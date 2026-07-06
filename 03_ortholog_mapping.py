# =============================================================================
# STEP 3: Orthologous Gene Mapping — Dog <-> Human
# Method: Ensembl FTP Compara homology files (static download, CanFam3.1 era)
#         + mygene.info for human Ensembl ID -> gene symbol lookup
# Maps canine Ensembl IDs (ENSCAFG*) to human gene symbols (matches Xena's
# symbol-indexed TCGA-BRCA matrix), then aligns both expression matrices
# on shared orthologous genes.
# =============================================================================

import pandas as pd
import numpy as np
import requests
import gzip
import io
import os

os.makedirs("data/aligned", exist_ok=True)

# =============================================================================
# Cell 1: Fetch ortholog table via Ensembl FTP Compara homology files
# =============================================================================
#
# WHY THIS APPROACH: The live BioMart archive web service
# (may2021/feb2021/nov2020.archive.ensembl.org) was returning HTML error
# pages instead of TSV data - a server-side outage on Ensembl's side.
# Instead we download Ensembl's STATIC precomputed Compara homology file
# for a CanFam3.1-era release via plain HTTPS - a normal file download,
# not a live database query, so it's far less affected by outages.

# CanFam3.1 was used up to and including release 104 (May 2021).
# Release 105 (Dec 2021) switched to the new ROS_Cfam_1.0 assembly.
FTP_RELEASES_TO_TRY = [104, 103, 102, 101, 100]

def _ftp_homology_url(release: int) -> str:
    return (
        f"https://ftp.ensembl.org/pub/release-{release}/tsv/"
        f"ensembl-compara/homologies/canis_lupus_familiaris/"
        f"Compara.{release}.protein_default.homologies.tsv.gz"
    )

def fetch_orthologs_biomart(batch_size=1000, max_genes=None):
    """
    Download the dog (canis_lupus_familiaris) Compara homology TSV from
    Ensembl's static FTP site for a CanFam3.1-era release, then filter down
    to dog<->human homologies. Returns a DataFrame with columns:
      dog_ensembl_id, dog_gene_symbol, human_ensembl_id, human_gene_symbol,
      orthology_type, pct_id_dog_to_human, pct_id_human_to_dog
    """
    last_error = None

    for release in FTP_RELEASES_TO_TRY:
        url = _ftp_homology_url(release)
        print(f"Downloading Ensembl Compara homology file: {url}")
        print("  (this file covers dog-vs-all-species homology; may take a minute) ...")
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  Download failed: {e} - trying next release ...")
            last_error = e
            continue

        try:
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                df = pd.read_csv(gz, sep="\t")
        except (gzip.BadGzipFile, OSError, pd.errors.ParserError) as e:
            print("  File wasn't valid gzip/TSV (probably a 404 HTML page) - trying next release ...")
            last_error = e
            continue

        print(f"  Downloaded OK. Raw shape: {df.shape}")

        required = {"gene_stable_id", "homology_species", "homology_gene_stable_id",
                    "homology_type", "identity", "homology_identity"}
        missing = required - set(df.columns)
        if missing:
            print(f"  Unexpected column layout, missing: {missing} - trying next release ...")
            last_error = RuntimeError(f"Unexpected columns from {url}: {df.columns.tolist()}")
            continue

        df_human = df[df["homology_species"] == "homo_sapiens"].copy()
        if df_human.empty:
            print("  No human homology rows found in this file - trying next release ...")
            last_error = RuntimeError(f"No human rows in {url}")
            continue

        print(f"  Human homology rows: {df_human.shape}")

        # Gene symbols aren't in this file; resolved later via mygene.info (Cell 2.5)
        out = pd.DataFrame({
            "dog_ensembl_id": df_human["gene_stable_id"],
            "dog_gene_symbol": np.nan,
            "human_ensembl_id": df_human["homology_gene_stable_id"],
            "human_gene_symbol": np.nan,
            "orthology_type": df_human["homology_type"],
            "pct_id_dog_to_human": df_human["identity"],
            "pct_id_human_to_dog": df_human["homology_identity"],
        })
        print(f"  SUCCESS from release {release}.")
        return out

    raise RuntimeError(
        "All Ensembl FTP Compara homology files failed to download/parse. "
        f"Last error: {last_error}\n"
        f"Try opening this URL in a browser to check manually: {_ftp_homology_url(104)}"
    )


# Check if cached ortholog table exists
ORTHOLOG_CACHE = "data/aligned/dog_human_orthologs_raw.csv"

def _cache_is_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        cached = pd.read_csv(path, index_col=0)
    except Exception:
        return False
    if cached.empty or cached.shape[1] != 7:
        return False
    if any("ERROR" in str(c).upper() or "DOCTYPE" in str(c).upper() or "HTML" in str(c).upper()
           for c in cached.columns):
        return False
    return True

if os.path.exists(ORTHOLOG_CACHE) and not _cache_is_valid(ORTHOLOG_CACHE):
    print(f"Discarding invalid cached file: {ORTHOLOG_CACHE}")
    os.remove(ORTHOLOG_CACHE)

if os.path.exists(ORTHOLOG_CACHE):
    print("Loading cached ortholog table ...")
    orthologs_raw = pd.read_csv(ORTHOLOG_CACHE, index_col=0)
    print(f"  Loaded: {orthologs_raw.shape}")
else:
    orthologs_raw = fetch_orthologs_biomart()
    orthologs_raw.to_csv(ORTHOLOG_CACHE)
    print(f"  Saved to cache: {ORTHOLOG_CACHE}")

print(f"\nRaw ortholog table:\n{orthologs_raw.head()}")

# =============================================================================
# Cell 2: Filter to HIGH-CONFIDENCE one-to-one orthologs only
# one2one = gene duplicated in neither species -> most reliable mapping
# Also filter by percent identity > 50% to ensure meaningful homology
# =============================================================================

print("\nColumns:", orthologs_raw.columns.tolist())

expected_cols = [
    "dog_ensembl_id", "dog_gene_symbol", "human_ensembl_id",
    "human_gene_symbol", "orthology_type",
    "pct_id_dog_to_human", "pct_id_human_to_dog"
]
if list(orthologs_raw.columns) != expected_cols and len(orthologs_raw.columns) == 7:
    orthologs_raw.columns = expected_cols

orthologs_clean = orthologs_raw.dropna(
    subset=["dog_ensembl_id", "human_ensembl_id"]
).copy()

print(f"\nAfter dropping NaN IDs: {orthologs_clean.shape}")

orthologs_1to1 = orthologs_clean[
    (orthologs_clean["orthology_type"] == "ortholog_one2one") &
    (orthologs_clean["pct_id_dog_to_human"] > 50) &
    (orthologs_clean["pct_id_human_to_dog"] > 50)
].copy()

print(f"After one2one + identity >50% filter: {orthologs_1to1.shape}")

orthologs_1to1 = orthologs_1to1.sort_values(
    "pct_id_dog_to_human", ascending=False
).drop_duplicates("dog_ensembl_id").drop_duplicates("human_ensembl_id")

print(f"After deduplication: {orthologs_1to1.shape}")
print(f"\nSample orthologs:")
print(orthologs_1to1.head(10).to_string(index=False))

orthologs_1to1.to_csv("data/aligned/dog_human_orthologs_filtered.csv", index=False)
print("\nSaved: data/aligned/dog_human_orthologs_filtered.csv")

# =============================================================================
# Cell 2.5: Fetch human gene SYMBOLS for our filtered ortholog IDs
# =============================================================================
#
# WHY: The TCGA-BRCA expression matrix from UCSC Xena (HiSeqV2_PANCAN) is
# indexed by gene SYMBOL (e.g. "TP53"), not Ensembl ID. Our ortholog table
# only has Ensembl IDs so far. Without symbols, remapping the canine matrix
# to Ensembl IDs will NEVER overlap with the symbol-indexed human matrix ->
# 0 shared genes. We fix that by looking up symbols via mygene.info (a
# standard, independent gene-annotation API - not part of Ensembl's
# infrastructure, so unaffected by the archive mirror issues).

def fetch_symbols_mygene(ensembl_ids, species="human", batch_size=1000):
    """Query mygene.info to map Ensembl gene IDs -> official gene symbol."""
    ensembl_ids = list(dict.fromkeys(ensembl_ids))  # dedupe, keep order
    symbol_map = {}
    for i in range(0, len(ensembl_ids), batch_size):
        batch = ensembl_ids[i:i + batch_size]
        print(f"  Querying mygene.info for symbols: {i+1}-{i+len(batch)} of {len(ensembl_ids)} ...")
        resp = requests.post(
            "https://mygene.info/v3/query",
            data={
                "q": ",".join(batch),
                "scopes": "ensembl.gene",
                "fields": "symbol",
                "species": species,
            },
            timeout=60,
        )
        resp.raise_for_status()
        for hit in resp.json():
            if "symbol" in hit and not hit.get("notfound"):
                symbol_map.setdefault(hit["query"], hit["symbol"])
    return symbol_map

print("\nFetching human gene symbols for filtered orthologs via mygene.info ...")
human_ids_needed = orthologs_1to1["human_ensembl_id"].unique().tolist()
ensg_to_symbol = fetch_symbols_mygene(human_ids_needed)
print(f"  Resolved symbols for {len(ensg_to_symbol)} / {len(human_ids_needed)} human Ensembl IDs")

orthologs_1to1["human_gene_symbol"] = orthologs_1to1["human_ensembl_id"].map(ensg_to_symbol)

n_before = len(orthologs_1to1)
orthologs_1to1 = orthologs_1to1.dropna(subset=["human_gene_symbol"]).copy()
print(f"  Orthologs with resolved symbol: {len(orthologs_1to1)} / {n_before}")

orthologs_1to1.to_csv("data/aligned/dog_human_orthologs_filtered.csv", index=False)
print("  Re-saved: data/aligned/dog_human_orthologs_filtered.csv (now with symbols)")

# =============================================================================
# Cell 3: Load both expression matrices
# =============================================================================

print("\nLoading expression matrices ...")
expr_human = pd.read_csv("data/human/TCGA_BRCA_tumor_normal_expression.csv", index_col=0)
expr_canine = pd.read_csv("data/canine/canine_fpkm_clean.csv", index_col=0)

print(f"  Human (genes x samples): {expr_human.shape}")
print(f"  Canine (genes x samples): {expr_canine.shape}")
print(f"  Human index sample: {expr_human.index[:5].tolist()}")

# =============================================================================
# Cell 4: Remap canine matrix — dog Ensembl ID -> human gene SYMBOL
# (matches the symbol-indexed Xena human matrix)
# =============================================================================

dog_to_human_sym = dict(zip(
    orthologs_1to1["dog_ensembl_id"],
    orthologs_1to1["human_gene_symbol"]
))

canine_genes_in_orthologs = [g for g in expr_canine.index if g in dog_to_human_sym]
print(f"\nCanine genes in expression matrix: {len(expr_canine)}")
print(f"Canine genes with human ortholog (symbol resolved): {len(canine_genes_in_orthologs)}")

expr_canine_mapped = expr_canine.loc[canine_genes_in_orthologs].copy()
expr_canine_mapped.index = [dog_to_human_sym[g] for g in expr_canine_mapped.index]

# If multiple dog genes mapped to the same human symbol, keep the first
expr_canine_mapped = expr_canine_mapped[~expr_canine_mapped.index.duplicated(keep="first")]

print(f"Canine matrix after remapping to human gene symbols: {expr_canine_mapped.shape}")

# =============================================================================
# Cell 5: Find shared genes between human and remapped canine matrix
# =============================================================================

human_genes = set(expr_human.index)
canine_human_genes = set(expr_canine_mapped.index)
shared_genes = sorted(human_genes & canine_human_genes)

print(f"\nHuman expression genes: {len(human_genes)}")
print(f"Canine ortholog-mapped genes: {len(canine_human_genes)}")
print(f"Shared orthologous genes: {len(shared_genes)}")

if len(shared_genes) == 0:
    raise RuntimeError(
        "Still 0 shared genes after symbol-based remapping. "
        "Print expr_human.index[:20] and list(canine_human_genes)[:20] "
        "to compare formatting (case, whitespace, versioned IDs, etc.)."
    )

expr_human_shared = expr_human.loc[shared_genes]
expr_canine_shared = expr_canine_mapped.loc[shared_genes]

print(f"\nFinal aligned shapes:")
print(f"  Human  (shared genes x samples): {expr_human_shared.shape}")
print(f"  Canine (shared genes x samples): {expr_canine_shared.shape}")

# =============================================================================
# Cell 6: Log2 normalisation
# =============================================================================

human_max = expr_human_shared.values.max()
print(f"\nHuman expression max value: {human_max:.2f}")
if human_max > 100:
    print("  Applying log2(x+1) to human data ...")
    expr_human_shared = np.log2(expr_human_shared + 1)
else:
    print("  Human data already log-transformed (Xena standard)")

canine_max = expr_canine_shared.values.max()
print(f"Canine expression max value: {canine_max:.2f}")
if canine_max > 100:
    print("  Applying log2(x+1) to canine data ...")
    expr_canine_shared = np.log2(expr_canine_shared + 1)
else:
    print("  Canine data already log-transformed")

# =============================================================================
# Cell 7: Save aligned matrices
# =============================================================================

expr_human_shared.to_csv("data/aligned/human_BRCA_aligned.csv")
expr_canine_shared.to_csv("data/aligned/canine_CMT_aligned.csv")

# Gene info now keyed by SYMBOL (since that's what the aligned matrices use)
gene_info = pd.DataFrame({
    "human_gene_symbol": shared_genes,
})
gene_info.to_csv("data/aligned/shared_ortholog_genes.csv", index=False)

print("\nSaved:")
print("  data/aligned/human_BRCA_aligned.csv")
print("  data/aligned/canine_CMT_aligned.csv")
print("  data/aligned/shared_ortholog_genes.csv")
print(f"\nShared gene count for ML: {len(shared_genes)}")
print("Step 3 COMPLETE.")
