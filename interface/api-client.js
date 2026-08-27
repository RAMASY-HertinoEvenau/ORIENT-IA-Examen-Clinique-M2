const URL_API = window.ORIENT_IA_API_URL !== undefined ? window.ORIENT_IA_API_URL : 'http://localhost:8000';

const reponseMock = {
  recommandations: [
    { parcours: 'Informatique de Gestion, Génie Logiciel et Intelligence Artificielle (IGGLIA)', pertinence: 82, pourquoi: ['Votre intérêt pour l’informatique', 'Votre curiosité pour la programmation', 'Un environnement de travail flexible'], prerequis: 'Baccalauréat et sélection de dossier.', debouches: 'Information non disponible dans les sources référencées.', incertitude: 'Modérée' },
    { parcours: 'Informatique Statistique Appliquée et Intelligence Artificielle (ISAIA)', pertinence: 67, pourquoi: ['Votre attrait pour les données', 'Votre intérêt pour les méthodes analytiques'], prerequis: 'Baccalauréat et sélection de dossier.', debouches: 'Banques, entreprises industrielles et entreprises commerciales.', incertitude: 'Modérée' }
  ],
  sources: [
    { nom: 'Les différents départements et filières', type: 'Source institutionnelle', origine: 'ISPM', url: 'http://www.ispm-edu.com/filieres.php', date: '26 août 2026', statut: 'Institutionnelle' },
    { nom: 'Conditions d’accès en première année', type: 'Source institutionnelle', origine: 'ISPM', url: 'http://www.ispm-edu.com/inscription.php', date: '26 août 2026', statut: 'Institutionnelle' }
  ],
  tracabilite: { question: 'Analyse du profil candidat', profil: 'Niveau, intérêts, compétences et préférences déclarés', outils: 'Analyse de profil', resultats: 'Classement synthétique de parcours' }
};

function attendre(delai) { return new Promise(resolve => setTimeout(resolve, delai)); }

export async function analyserProfil(profil) {
  if (URL_API) {
    try {
      const response = await fetch(`${URL_API}/profil/analyser`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(profil) });
      if (response.ok) return response.json();
    } catch (e) {
      console.warn("Backend local non joignable, bascule vers mode local/simulation.");
    }
  }
  await attendre(600);
  if (!profil.niveau) return { recommandations: [], sources: [], erreur: 'profil_incomplet' };
  return structuredClone(reponseMock);
}

export async function envoyerMessage(message, profil) {
  if (URL_API) {
    try {
      const response = await fetch(`${URL_API}/agent/message`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, profil }) });
      if (response.ok) return response.json();
    } catch (e) {
      console.warn("Backend local non joignable, bascule vers mode local/simulation.");
    }
  }
  await attendre(400);
  return { message: 'À partir des informations fournies, les parcours affichés sont les pistes les plus pertinentes dans le périmètre des sources référencées. Je peux préciser la comparaison si vous me dites ce qui compte le plus pour vous.', tracabilite: reponseMock.tracabilite };
}

export const modeDemonstration = false;
