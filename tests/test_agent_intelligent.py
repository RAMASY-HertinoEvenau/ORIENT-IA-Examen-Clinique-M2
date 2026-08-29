from orient_ia.agent.orchestrateur import AgentOrientIA
from orient_ia.agent.parcours import extraire_codes_parcours


def test_extraction_tee_teh() -> None:
    assert extraire_codes_parcours("Quelle est la différence entre TEE et TEH ?") == [
        "parcours-tee",
        "parcours-teh",
    ]


def test_comparaison_tee_teh_ne_parle_pas_d_esiia() -> None:
    agent = AgentOrientIA()
    res = agent.traiter_message("Quelle est la différence entre TEE et TEH ?")
    texte = res["reponse"].lower()
    assert "tourisme et environnement" in texte
    assert "hôtellerie" in texte or "hotellerie" in texte
    assert "esiia" not in texte
    assert "électronique" not in texte or "écotourisme" in texte


def test_metiers_gca() -> None:
    agent = AgentOrientIA()
    res = agent.traiter_message("Quels métiers après GCA ?")
    texte = res["reponse"].lower()
    assert "génie civil" in texte or "gca" in texte
    assert "travaux" in texte or "structure" in texte or "chantier" in texte


def test_debouches_igglia() -> None:
    agent = AgentOrientIA()
    res = agent.traiter_message("Quels sont les débouchés après IGGLIA ?")
    texte = res["reponse"].lower()
    assert "igglia" in texte
    assert "développeur" in texte or "logiciel" in texte
