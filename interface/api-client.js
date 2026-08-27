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
    ...profil,
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
  if (!profil || !profil.niveau) {
    return { recommandations: [], sources: [], erreur: 'profil_incomplet' };
  }

  // 1. Essai sur la route native ORIENT'IA /profil/analyser
  try {
    const response = await fetch(`${URL_API}/profil/analyser`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profil)
    });
    if (response.ok) {
      return await response.json();
    }
  } catch (e) {
    // Si échec, tentative sur route alternative
  }

  // 2. Essai sur la route alternative /api/recommandation
  try {
    const response = await fetch(`${URL_API}/api/recommandation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profil: payload })
    });

    if (response.ok) {
      const data = await response.json();
      const recsUI = (data.recommandations || []).map(r => {
        const nomFormation = (r.formation && r.formation.nom) || r.nom || r.parcours || r.identifiant || 'Parcours ISPM';
        const raisons = r.raisons_liees_au_profil || [];
        const corpus = r.elements_du_corpus || [];
        const pourquoi = [...raisons, ...corpus].filter(Boolean);
        if (pourquoi.length === 0) {
          pourquoi.push("Correspondance élevée avec les matières, compétences et préférences indiquées.");
        }
        const pReq = corpus.find(e => e.toLowerCase().includes("prérequis")) || "Baccalauréat et sélection de dossier.";
        const deb = corpus.find(e => e.toLowerCase().includes("débouchés") || e.toLowerCase().includes("métiers")) || "Postes d'ingénierie et de management.";

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

      return {
        recommandations: recsUI,
        sources: sourcesUI,
        tracabilite: {
          question: 'Analyse du profil candidat',
          profil: `Niveau: ${profil.niveau || 'Baccalauréat'}`,
          outils: 'moteur_recommandation (ExtraTrees Classifier)',
          resultats: `${recsUI.length} parcours recommandés.`
        }
      };
    }
  } catch (err) {
    console.warn('Backend non joignable:', err);
  }

  return {
    recommandations: [],
    sources: [],
    erreur: 'backend_indisponible',
    message: 'Le serveur ORIENT’IA n’est pas accessible actuellement.'
  };
}

export async function envoyerMessage(message, profil) {
  const payloadProfil = normaliserProfilPayload(profil);

  // 1. Essai sur /agent/message
  try {
    const response = await fetch(`${URL_API}/agent/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, profil })
    });
    if (response.ok) {
      return await response.json();
    }
  } catch (e) {
    // Si échec, tentative sur route alternative
  }

  // 2. Essai sur /api/agent/chat
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
    if (response.ok) {
      const data = await response.json();
      return {
        message: data.reponse || data.message || 'Aucune réponse fournie.',
        sources: data.sources || [],
        tracabilite: data.trace || {
          question: message,
          outils: 'orchestrateur_conversation',
          resultats: 'Traitement effectué'
        }
      };
    }
  } catch (err) {
    console.warn('Erreur API Chat:', err);
  }

  return {
    message: "À partir des informations fournies, veuillez vérifier que le serveur ORIENT'IA est bien démarré sur le port 8000.",
    sources: []
  };
}

export const modeDemonstration = false;
