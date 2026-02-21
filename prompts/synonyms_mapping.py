# =============================================================================
# DIAGNOSIS SYNONYMS MAPPING FOR VERIFICATION
# =============================================================================
# This configuration maps various synonyms and abbreviations to canonical
# SNOMED CT terms for use during extraction verification.
#
# Usage: When comparing extracted diagnosis names with ground truth,
#        use this mapping to normalize terms before comparison.
# =============================================================================

DIAGNOSIS_SYNONYMS = {
    # Heart Failure variants
    "Heart Failure": [
        "CHF", "Congestive Heart Failure", "HF", "Heart failure",
        "Cardiac Failure", "cardiac failure", "congestive heart failure"
    ],
    
    # COPD variants
    "COPD": [
        "Chronic Obstructive Pulmonary Disease", "chronic obstructive pulmonary disease",
        "Chronic lung disease", "Emphysema", "Chronic bronchitis"
    ],
    
    # Chronic Kidney Disease variants
    "Chronic Kidney Disease": [
        "CKD", "Renal failure", "Kidney disease", "chronic kidney disease",
        "Renal insufficiency", "End stage renal disease", "ESRD", "Nephropathy"
    ],
    
    # Diabetes variants
    "Diabetes Mellitus": [
        "DM", "DM2", "DMII", "Diabetes", "Type 2 diabetes", "Type 2 diabetes mellitus",
        "T2DM", "Diabetes type 2", "diabetes mellitus", "Type II diabetes"
    ],
    
    # Hypertension variants
    "Hypertension": [
        "HTN", "High blood pressure", "Essential hypertension", "hypertension",
        "Elevated blood pressure", "HBP"
    ],
    
    # Cancer variants
    "Cancer": [
        "Malignancy", "Neoplasm", "Carcinoma", "Tumor", "Metastatic cancer",
        "Oncology", "CA", "cancer"
    ],
    
    # Dementia variants
    "Dementia": [
        "Alzheimer's disease", "Alzheimers", "Cognitive impairment",
        "Memory loss", "dementia", "Senile dementia"
    ],
    
    # Cardiac Arrhythmias variants
    "Cardiac Arrhythmias": [
        "Arrhythmia", "Atrial fibrillation", "AFib", "AF", "A-fib",
        "Irregular heartbeat", "arrhythmias", "Ventricular arrhythmia"
    ],
    
    # Stroke variants
    "Stroke": [
        "CVA", "Cerebrovascular accident", "stroke", "TIA",
        "Transient ischemic attack", "Brain attack", "Ischemic stroke"
    ],
    
    # Peripheral Vascular Disease variants
    "Peripheral Vascular Disease": [
        "PVD", "PAD", "Peripheral arterial disease", "peripheral vascular disease",
        "Claudication", "Peripheral artery disease"
    ],
    
    # Hyperlipidemia variants
    "Hyperlipidemia": [
        "HLD", "High cholesterol", "Dyslipidemia", "hyperlipidemia",
        "Elevated lipids", "Hypercholesterolemia"
    ]
}

# =============================================================================
# SYMPTOM SYNONYMS MAPPING
# =============================================================================

SYMPTOM_SYNONYMS = {
    "Nausea": ["nausea", "Nauseated", "nauseated", "N"],
    "Vomiting": ["vomiting", "Emesis", "V", "N/V"],
    "Dizziness": ["dizziness", "Dizzy", "Light-headed", "Lightheadedness"],
    "Headache": ["headache", "HA", "Head pain", "Cephalalgia"],
    "Diarrhea": ["diarrhea", "Loose stools", "Watery stool"],
    "Vertigo": ["vertigo", "Room spinning", "Spinning sensation"],
    "Chest pain": ["chest pain", "CP", "Chest discomfort"],
    "Shortness of breath": ["SOB", "Dyspnea", "Breathlessness", "DOE"],
}

# =============================================================================
# MEDICATION FLAG SYNONYMS
# =============================================================================

MEDICATION_SYNONYMS = {
    "Anticoagulation": ["anticoagulation", "Blood thinners", "Warfarin", "Coumadin", "Eliquis", "Xarelto", "Heparin"],
    "Insulin Therapy": ["Insulin", "insulin therapy", "Lantus", "Humalog", "Novolog"],
    "Opioid Therapy": ["Opioid", "Oxycodone", "Hydrocodone", "Morphine", "Fentanyl", "opioid therapy"],
    "Diuretic Therapy": ["Diuretic", "Lasix", "Furosemide", "HCTZ", "diuretic therapy"],
}

# =============================================================================
# SEX/DEMOGRAPHICS SYNONYMS MAPPING
# =============================================================================

SEX_SYNONYMS = {
    "male": ["M", "m", "Male", "MALE", "Man", "man"],
    "female": ["F", "f", "Female", "FEMALE", "Woman", "woman"],
}

def normalize_sex(extracted_term: str) -> str:
    """Normalize sex value to canonical form (male/female)."""
    extracted_clean = extracted_term.strip()
    
    for canonical, synonyms in SEX_SYNONYMS.items():
        if extracted_clean.lower() == canonical:
            return canonical
        for syn in synonyms:
            if extracted_clean == syn:
                return canonical
    
    return extracted_term


# =============================================================================
# HELPER FUNCTIONS FOR VERIFICATION
# =============================================================================

def normalize_diagnosis(extracted_term: str) -> str:
    """
    Normalize an extracted diagnosis term to its canonical form.
    
    Args:
        extracted_term: The term extracted by the model
        
    Returns:
        Canonical SNOMED CT term, or original if no match found
    """
    extracted_lower = extracted_term.lower().strip()
    
    for canonical, synonyms in DIAGNOSIS_SYNONYMS.items():
        if extracted_lower == canonical.lower():
            return canonical
        for syn in synonyms:
            if extracted_lower == syn.lower():
                return canonical
    
    return extracted_term  # Return original if no match


def normalize_symptom(extracted_term: str) -> str:
    """Normalize an extracted symptom term to its canonical form."""
    extracted_lower = extracted_term.lower().strip()
    
    for canonical, synonyms in SYMPTOM_SYNONYMS.items():
        if extracted_lower == canonical.lower():
            return canonical
        for syn in synonyms:
            if extracted_lower == syn.lower():
                return canonical
    
    return extracted_term


def terms_match(extracted: str, ground_truth: str, category: str = "diagnosis") -> bool:
    """
    Check if extracted term matches ground truth, considering synonyms.
    
    Args:
        extracted: Term extracted by the model
        ground_truth: Term from gold standard
        category: One of "diagnosis", "symptom", "medication"
        
    Returns:
        True if terms match (exact or via synonyms)
    """
    if category == "diagnosis":
        return normalize_diagnosis(extracted) == normalize_diagnosis(ground_truth)
    elif category == "symptom":
        return normalize_symptom(extracted) == normalize_symptom(ground_truth)
    else:
        return extracted.lower().strip() == ground_truth.lower().strip()


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'DIAGNOSIS_SYNONYMS',
    'SYMPTOM_SYNONYMS', 
    'MEDICATION_SYNONYMS',
    'normalize_diagnosis',
    'normalize_symptom',
    'terms_match'
]
