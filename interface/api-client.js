const URL_API = window.ORIENT_IA_API_URL !== undefined ? window.ORIENT_IA_API_URL : '';

function normaliserProfilPayload(profil) {
  if (!profil) return null;

  let matieres = profil.matieres_preferees;
  if (typeof matieres === 'string') {
    matieres = matieres.split(',').map(s => s.trim()).filter(Boolean);
  }
  if (!Array.isArray(matieres) || matieres.length === 0) {
    matieres = ["informatique", "mathematiques"];
  }

  const moyenne = parseFloat(profil.moyenne_scolaire || 15.0);

  return {
    matieres_preferees: matieres,
    moyenne_scolaire: isNaN(moyenne) ? 15.0 : moyenne,
    competences: {
      "competence-techniques-informatiques-gestion": 4,
      "competence-electronique-systemes": 3
    },
    centres_interet: Array.isArray(profil.centres_interet) ? profil.centres_interet : [],
    projets: Array.isArray(profil.projets) ? profil.projets : (profil.projets ? [profil.projets] : []),
    preferences_professionnelles: profil.preferences_professionnelles || "salariat",
    environnement_travail: profil.environnement_travail || "hybride"
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

    const recsUI = (data.recommandations || []).map(r => ({
      parcours: r.nom || r.parcours || r.identifiant || 'Parcours conseillé',
      pertinence: Math.round((r.score !== undefined ? r.score : 0.8) * 100),
      pourquoi: r.limites && r.limites.length ? r.limites : ['Correspondance avec les matières et compétences indiquées.'],
      prerequis: r.prerequis ? r.prerequis.join(', ') : 'Baccalauréat et sélection de dossier.',
      debouches: r.metiers ? r.metiers.join(', ') : 'Ingénierie, gestion, services et entrepreneuriat.',
      incertitude: r.incertitude || data.incertitude || 'Prudent et indicatif'
    }));

    const sourcesUI = (data.sources || []).map(s => ({
      nom: typeof s === 'string' ? s : (s.titre || s.identifiant || 'Source ISPM'),
      type: typeof s === 'object' && s.statut ? s.statut : 'Corpus institutionnel',
      origine: typeof s === 'object' && s.origine ? s.origine : 'ISPM',
      url: typeof s === 'object' ? s.url : 'http://www.ispm-edu.com',
      date: typeof s === 'object' ? s.date_consultation : '2026-08-26',
      statut: 'Institutionnelle'
    }));

    const tracabiliteUI = {
      question: 'Analyse et recommandation du profil candidat',
      profil: `Matières: ${(payload.matieres_preferees || []).join(', ')}, Moyenne: ${payload.moyenne_scolaire}`,
      outils: 'moteur_recommandation (analyser_profil)',
      resultats: `Statut: ${data.status || 'ok'}, ${recsUI.length} piste(s) proposée(s)`
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
    const traceData = data.trace || {};

    const sourcesUI = (data.sources || []).map(s => ({
      nom: typeof s === 'string' ? s : (s.titre || s.identifiant || 'Document RAG'),
      type: 'Corpus RAG',
      origine: 'ISPM',
      url: typeof s === 'object' ? s.url : '',
      date: '2026-08-26',
      statut: 'Institutionnelle'
    }));

    const tracabiliteUI = {
      question: message,
      profil: payloadProfil ? `Matières: ${(payloadProfil.matieres_preferees || []).join(', ')}` : 'Non spécifié',
      outils: (data.outils_appeles || []).join(', ') || 'orchestrateur_conversation',
      resultats: `État: ${data.etat}, ${sourcesUI.length} source(s) liée(s)`
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
