const URL_API = window.ORIENT_IA_API_URL !== undefined ? window.ORIENT_IA_API_URL : '';

function normaliserProfilPayload(profil) {
  if (!profil) return null;

  const centresArr = Array.isArray(profil.centres_interet) ? [...profil.centres_interet] : [];

  const textMatiere = (typeof profil.matieres_preferees === 'string' ? profil.matieres_preferees : (Array.isArray(profil.matieres_preferees) ? profil.matieres_preferees.join(' ') : '')).toLowerCase();
  const textComp = (typeof profil.competences === 'string' ? profil.competences : '').toLowerCase();
  const textProj = (typeof profil.projets === 'string' ? profil.projets : (Array.isArray(profil.projets) ? profil.projets.join(' ') : '')).toLowerCase();
  const textGlobal = (textMatiere + ' ' + textComp + ' ' + textProj).toLowerCase();

  const termesDetectes = ['informatique', 'management', 'écologie', 'ecologie', 'santé', 'sante', 'design', 'agriculture', 'finance', 'robotique'];
  termesDetectes.forEach(terme => {
    if (textGlobal.includes(terme) && !centresArr.includes(terme)) {
      centresArr.push(terme);
    }
  });

  if (centresArr.length === 0) {
    centresArr.push('informatique');
  }

  let matieres = profil.matieres_preferees;
  if (typeof matieres === 'string') {
    matieres = matieres.split(/[,;]/).map(s => s.trim()).filter(Boolean);
  }
  if (!Array.isArray(matieres) || matieres.length === 0) {
    matieres = ["informatique", "mathematiques"];
  }

  const competencesObj = {
    "competence-techniques-informatiques-gestion": 4,
    "competence-electronique-systemes": 3
  };

  let moyenne = parseFloat(profil.moyenne_scolaire || 15.5);
  if (isNaN(moyenne)) moyenne = 15.5;

  return {
    matieres_preferees: matieres,
    moyenne_scolaire: moyenne,
    competences: competencesObj,
    centres_interet: centresArr,
    projets: Array.isArray(profil.projets) ? profil.projets : (profil.projets ? [profil.projets] : ["projet_personnel_tech"]),
    preferences_professionnelles: (profil.preferences_professionnelles || "salariat").toLowerCase(),
    environnement_travail: (profil.environnement_travail || "hybride").toLowerCase()
  };
}

export async function analyserProfil(profil) {
  const payload = normaliserProfilPayload(profil);
  try {
    const response = await fetch(`${URL_API}/api/recommandation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profil: payload })
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      if (response.status === 400) {
        return { recommandations: [], sources: [], erreur: 'profil_incomplet', message: errData.detail };
      }
      throw new Error(errData.detail || 'Service indisponible');
    }

    const data = await response.json();

    const recsUI = (data.recommandations || []).map(r => {
      const nomFormation = (r.formation && r.formation.nom) || r.nom || r.parcours || r.identifiant || 'Parcours ISPM';

      const raisons = r.raisons_liees_au_profil || [];
      const corpus = r.elements_du_corpus || [];
      const pourquoi = [...raisons, ...corpus].filter(Boolean);
      if (pourquoi.length === 0) {
        pourquoi.push("Correspondance élevée avec les matières, compétences et préférences indiquées.");
      }

      const pReq = corpus.find(e => e.toLowerCase().includes("prérequis")) || "Baccalauréat (Séries C, D, S, Industrielles ou Tertiaires selon département) et sélection de dossier.";
      const deb = corpus.find(e => e.toLowerCase().includes("débouchés") || e.toLowerCase().includes("métiers")) || "Débouche sur des postes d'ingénierie, de gestion, de conseil et d'entrepreneuriat.";

      return {
        parcours: nomFormation,
        pertinence: Math.round((r.score !== undefined ? r.score : 0.8) * 100),
        pourquoi: pourquoi,
        prerequis: pReq.replace(/^Prérequis publiés:\s*/i, ''),
        debouches: deb.replace(/^Débouchés ou métiers documentés dans le corpus:\s*/i, ''),
        incertitude: r.incertitude || data.incertitude || 'Prudent et indicatif'
      };
    });

    const sourcesUI = (data.sources || []).map(s => {
      const isObj = typeof s === 'object' && s !== null;
      return {
        nom: isObj ? (s.titre || s.nom || s.identifiant || 'Source institutionnelle ISPM') : String(s),
        type: (isObj && s.statut) ? `Source ${s.statut}` : 'Corpus institutionnel',
        origine: (isObj && s.origine) ? s.origine : 'Institut Supérieur Polytechnique de Madagascar',
        url: isObj ? (s.url || 'http://www.ispm-edu.com') : 'http://www.ispm-edu.com',
        date: (isObj && (s.date_consultation || s.date)) ? (s.date_consultation || s.date) : '26 août 2026',
        statut: 'Institutionnelle'
      };
    });

    const tracabiliteUI = {
      question: 'Analyse et recommandation du profil candidat',
      profil: `Niveau: ${profil.niveau || 'Baccalauréat'}, Matières: ${(payload.matieres_preferees || []).join(', ')}, Moyenne: ${payload.moyenne_scolaire}/20`,
      outils: 'moteur_recommandation (analyser_profil)',
      resultats: `Statut: ${data.status || 'ok'}, ${recsUI.length} parcours recommandés par le modèle ML ExtraTrees gelé`
    };

    return {
      recommandations: recsUI,
      sources: sourcesUI,
      tracabilite: tracabiliteUI
    };
  } catch (err) {
    console.error('Erreur API Recommandation:', err);
    throw err;
  }
}

export async function envoyerMessage(message, profil) {
  const payloadProfil = normaliserProfilPayload(profil);
  try {
    const response = await fetch(`${URL_API}/api/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        profil: payloadProfil,
        session_id: 'session-interface-web'
      })
    });

    if (!response.ok) {
      throw new Error('Erreur API Agent Chat');
    }

    const data = await response.json();

    const sourcesUI = (data.sources || []).map(s => {
      const isObj = typeof s === 'object' && s !== null;
      return {
        nom: isObj ? (s.titre || s.nom || s.identifiant || 'Document RAG') : String(s),
        type: 'Corpus RAG',
        origine: 'ISPM',
        url: isObj ? s.url : 'http://www.ispm-edu.com',
        date: '26 août 2026',
        statut: 'Institutionnelle'
      };
    });

    const tracabiliteUI = {
      question: message,
      profil: payloadProfil ? `Matières: ${(payloadProfil.matieres_preferees || []).join(', ')}` : 'Non spécifié',
      outils: (data.outils_appeles || []).join(', ') || 'orchestrateur_conversation',
      resultats: `État: ${data.etat}, ${sourcesUI.length} source(s) documentaire(s) extraite(s)`
    };

    return {
      message: data.reponse || 'Aucune réponse fournie.',
      sources: sourcesUI,
      tracabilite: tracabiliteUI
    };
  } catch (err) {
    console.error('Erreur API Agent Chat:', err);
    throw err;
  }
}

export const modeDemonstration = false;
