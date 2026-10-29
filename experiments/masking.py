"""Masks the persona labels of the prompt (profession, gender label, nationality) before LLM rating, so that the rater
judges the portrayal and not the label. Names, pronouns, places and tasks are left unchanged."""
import re

PATTERNS = [
    (r"\b(janitors?|janitorial|custodians?|custodial|engineers?|doctors?|physicians?|nurses?|teachers?)\b", "[ROLE]"),
    (r"\b(non-binary|nonbinary|non binary|male|female)\b", "[GENDER]"),
    (r"\b(american|german|indian|nigerian|chinese|americans|germans|indians|nigerians)\b", "[NATIONALITY]"),
    (r"\b(united states|usa|u\.s\.|america|germany|india|nigeria|china)\b", "[COUNTRY]"),
]

def mask(text):
    for pat, rep in PATTERNS:
        text = re.sub(pat, rep, text, flags=re.I)
    return text

# Experiment 4 uses more occupations: their labels are masked in the same way.
EXTRA_ROLES = (r"\b(police officers?|officers?|policem[ae]n|policewom[ae]n|firefighters?|fire ?fighters?|bankers?|judges?|"
               r"politicians?|farmers?|craftspeople|craftspersons?|crafts(?:man|men|woman|women)|childcare workers?|"
               r"child care workers?|childcare|daycare)\b")

def mask_ext(text):
    return mask(re.sub(EXTRA_ROLES, "[ROLE]", text, flags=re.I))
