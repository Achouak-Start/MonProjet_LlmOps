from transformers import AutoModelForCausalLM, AutoTokenizer

NOM_MODELE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"


def charger_modele_et_tokeniseur():
    """Charge le tokeniseur et le modèle LLM une seule fois."""
    tokeniseur = AutoTokenizer.from_pretrained(NOM_MODELE_LLM)
    modele = AutoModelForCausalLM.from_pretrained(NOM_MODELE_LLM)
    return modele, tokeniseur


def generer_reponse(
    prompt: str, modele, tokeniseur, max_nouveaux_tokens: int = 200
) -> str:
    """Génère une réponse à partir d'un prompt, en utilisant le format de chat du modèle."""
    messages = [{"role": "user", "content": prompt}]
    texte_formate = tokeniseur.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    entrees = tokeniseur(texte_formate, return_tensors="pt")

    sortie = modele.generate(
        **entrees,
        max_new_tokens=max_nouveaux_tokens,
        do_sample=False,
    )

    # On ne garde que les nouveaux tokens générés (pas le prompt répété)
    nouveaux_tokens = sortie[0][entrees["input_ids"].shape[1] :]
    reponse = tokeniseur.decode(nouveaux_tokens, skip_special_tokens=True)
    return reponse.strip()
