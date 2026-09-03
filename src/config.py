"""
PRAMAAN AI — Central Configuration Module
Single source of truth for all tunable thresholds, weights, and pipeline parameters.
Import from here instead of hardcoding values in engine files.
"""


# =============================================================================
# RISK SCORING & DECISION THRESHOLDS
# =============================================================================
AUTO_FLAG_RISK = 45.0           # Composite risk (0–100) at which a case is auto-flagged
FACE_MATCH_MINIMUM = 0.85       # Minimum biometric similarity for auto-clear

# =============================================================================
# TAMPERING ENGINE — ELA
# =============================================================================
ELA_QUALITY = 90                # JPEG recompression quality for ELA
ELA_MULTIPLIER = 15.0           # Brightness enhancement multiplier
ELA_MEAN_ERROR_DIVISOR = 45.0   # Normalization divisor for mean ELA error
ELA_ANOMALY_WEIGHT = 0.15       # Per-anomaly score contribution
ELA_GRID_SIZE = 32              # Grid cell size for anomaly scanning
ELA_THRESHOLD_STD = 2.2         # Standard deviations for anomaly threshold
ELA_MIN_INTENSITY = 25.0        # Minimum cell mean to be considered anomalous

# =============================================================================
# TAMPERING ENGINE — NOISE
# =============================================================================
NOISE_BLOCK_SIZE = 24           # Block size for noise variance calculation
NOISE_Z_SCORE_THRESHOLD = 3.0   # Z-score threshold for noise anomaly detection
NOISE_CV_DIVISOR = 1.5          # Coefficient-of-variation divisor for score normalization

# =============================================================================
# TAMPERING ENGINE — SPLICING
# =============================================================================
SPLICING_EDGE_DIVISOR = 80.0    # Normalization divisor for border gradient anomaly
SPLICING_DETECTION_THRESHOLD = 0.45  # Score above which splicing is flagged

# =============================================================================
# TAMPERING ENGINE — FONT FORENSICS
# =============================================================================
FONT_HEIGHT_DEVIATION_PX = 12   # Pixel threshold for height anomaly
FONT_HEIGHT_RATIO = 0.4         # Ratio of mean height for anomaly trigger
FONT_BASELINE_JITTER_PX = 10    # Pixel threshold for baseline jitter
FONT_ANOMALY_SCORE_PER_TOKEN = 0.25  # Per-flagged-token score contribution
FONT_MIN_TOKENS = 3             # Minimum tokens needed for analysis
FONT_LINE_CLUSTER_PX = 15       # Vertical distance threshold for line grouping

# =============================================================================
# TAMPERING ENGINE — METADATA
# =============================================================================
METADATA_SCAN_BYTES_HEAD = 16384  # Bytes to scan from file head (16KB)
METADATA_SCAN_BYTES_TAIL = 4096   # Bytes to scan from file tail (4KB)

# =============================================================================
# TAMPERING ENGINE — ENSEMBLE WEIGHTS
# =============================================================================
TAMPER_WEIGHT_ELA = 0.30
TAMPER_WEIGHT_NOISE = 0.25
TAMPER_WEIGHT_SPLICING = 0.25
TAMPER_WEIGHT_FONT = 0.15
TAMPER_WEIGHT_METADATA = 0.05

# Risk level thresholds (percentage)
TAMPER_THRESHOLD_CLEAN = 25.0
TAMPER_THRESHOLD_LOW_RISK = 45.0
TAMPER_THRESHOLD_HIGH_RISK = 70.0

# =============================================================================
# FACE ENGINE
# =============================================================================
FACE_SIMILARITY_LOW = 0.35      # Cosine similarity lower bound for calibration
FACE_SIMILARITY_RANGE = 0.50    # Range for calibrated normalization
FACE_MATCH_THRESHOLD = 0.65     # Default threshold for match verdict
FACE_MIN_AREA_RATIO = 0.015     # Minimum face area relative to image
FACE_ASPECT_RATIO_MIN = 0.8     # Minimum aspect ratio for face contour
FACE_ASPECT_RATIO_MAX = 2.5     # Maximum aspect ratio for face contour

# Liveness thresholds
LIVENESS_SHARPNESS_THRESHOLD = 30.0
LIVENESS_SATURATION_MIN = 15
LIVENESS_SATURATION_MAX = 200

# =============================================================================
# OVERALL RISK SCORING WEIGHTS
# =============================================================================
OVERALL_WEIGHT_TAMPERING = 0.50
OVERALL_WEIGHT_VALIDATION = 0.35
OVERALL_WEIGHT_OCR = 0.15

# =============================================================================
# SERVER & SECURITY
# =============================================================================
MAX_UPLOAD_SIZE_MB = 15         # Maximum file upload size
MAX_BACKGROUND_JOBS = 100       # Maximum concurrent background job records
BACKGROUND_JOB_TTL_SECONDS = 3600  # 1 hour TTL for completed background jobs

# =============================================================================
# VALIDATION ENGINE
# =============================================================================
# Full ISO 3166-1 alpha-3 country codes (249 countries)
ISO_3166_ALPHA3_CODES = {
    "AFG", "ALB", "DZA", "ASM", "AND", "AGO", "AIA", "ATA", "ATG", "ARG",
    "ARM", "ABW", "AUS", "AUT", "AZE", "BHS", "BHR", "BGD", "BRB", "BLR",
    "BEL", "BLZ", "BEN", "BMU", "BTN", "BOL", "BES", "BIH", "BWA", "BVT",
    "BRA", "IOT", "BRN", "BGR", "BFA", "BDI", "CPV", "KHM", "CMR", "CAN",
    "CYM", "CAF", "TCD", "CHL", "CHN", "CXR", "CCK", "COL", "COM", "COD",
    "COG", "COK", "CRI", "HRV", "CUB", "CUW", "CYP", "CZE", "CIV", "DNK",
    "DJI", "DMA", "DOM", "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ",
    "ETH", "FLK", "FRO", "FJI", "FIN", "FRA", "GUF", "PYF", "ATF", "GAB",
    "GMB", "GEO", "DEU", "GHA", "GIB", "GRC", "GRL", "GRD", "GLP", "GUM",
    "GTM", "GGY", "GIN", "GNB", "GUY", "HTI", "HMD", "VAT", "HND", "HKG",
    "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", "IRL", "IMN", "ISR", "ITA",
    "JAM", "JPN", "JEY", "JOR", "KAZ", "KEN", "KIR", "PRK", "KOR", "KWT",
    "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX",
    "MAC", "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MTQ", "MRT",
    "MUS", "MYT", "MEX", "FSM", "MDA", "MCO", "MNG", "MNE", "MSR", "MAR",
    "MOZ", "MMR", "NAM", "NRU", "NPL", "NLD", "NCL", "NZL", "NIC", "NER",
    "NGA", "NIU", "NFK", "MKD", "MNP", "NOR", "OMN", "PAK", "PLW", "PSE",
    "PAN", "PNG", "PRY", "PER", "PHL", "PCN", "POL", "PRT", "PRI", "QAT",
    "ROU", "RUS", "RWA", "REU", "BLM", "SHN", "KNA", "LCA", "MAF", "SPM",
    "VCT", "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", "SGP",
    "SXM", "SVK", "SVN", "SLB", "SOM", "ZAF", "SGS", "SSD", "ESP", "LKA",
    "SDN", "SUR", "SJM", "SWE", "CHE", "SYR", "TWN", "TJK", "TZA", "THA",
    "TLS", "TGO", "TKL", "TON", "TTO", "TUN", "TUR", "TKM", "TCA", "TUV",
    "UGA", "UKR", "ARE", "GBR", "UMI", "USA", "URY", "UZB", "VUT", "VEN",
    "VNM", "VGB", "VIR", "WLF", "ESH", "YEM", "ZMB", "ZWE", "ALA",
    # ICAO special codes
    "UNO", "UNA", "UNK", "D", "EUE", "XXA", "XXB", "XXC", "XXX",
}
