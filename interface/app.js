import { analyserProfil, envoyerMessage, modeDemonstration } from './api-client.js';

const form = document.querySelector('#profile-form');
const chatForm = document.querySelector('#chat-form');
const chatLog = document.querySelector('#chat-log');
const recommendations = document.querySelector('#recommendations');
const sourcesList = document.querySelector('#sources-list');
const resultCount = document.querySelector('#result-count');
const demoNotice = document.querySelector('#demo-notice');
const connectionLabel = document.querySelector('#connection-label');
const traceContent = document.querySelector('#trace-content');
let profilCourant = {};

const valeursCochees = nom => [...document.querySelectorAll(`input[name="${nom}"]:checked`)].map(input => input.value);
const lireProfil = () => ({
  niveau: document.querySelector('#niveau').value,
  matieres_preferees: document.querySelector('#matieres').value,
  competences: document.querySelector('#competences').value,
  centres_interet: valeursCochees('centres_interet'),
  projets: document.querySelector('#projets').value,
  preferences_professionnelles: document.querySelector('#preference').value,
  environnement_travail: document.querySelector('#environnement').value
});

function afficherEtat(message, type = 'assistant') {
  const element = document.createElement('div');
  element.className = `message ${type === 'user' ? 'user-message' : 'assistant-message'}`;
  element.innerHTML = type === 'user'
    ? `<div class="message-body"><p>${echapper(message)}</p></div><div class="avatar">V</div>`
    : `<div class="avatar">O</div><div><span class="message-author">ORIENT’IA <time>maintenant</time></span><p>${echapper(message)}</p></div>`;
  chatLog.appendChild(element);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function echapper(value) {
  const div = document.createElement('div');
  div.textContent = value;
  return div.innerHTML;
}

function afficherChargement() {
  resultCount.textContent = 'Analyse en cours';
  recommendations.innerHTML = '<div class="recommendations-empty"><div class="empty-symbol" aria-hidden="true">◌</div><h3>Analyse du profil en cours...</h3><p>Les informations sont transmises au service ORIENT’IA.</p></div>';
}

function afficherRecommandations(resultat) {
  if (!resultat || resultat.erreur === 'profil_incomplet') {
    resultCount.textContent = 'Profil incomplet';
    recommendations.innerHTML = '<div class="recommendations-empty"><div class="empty-symbol" aria-hidden="true">!</div><h3>Certaines informations sont nécessaires.</h3><p>Indiquez au moins votre niveau d’étude pour améliorer la recommandation.</p></div>';
    return;
  }
  const items = resultat.recommandations || [];
  resultCount.textContent = `${items.length} piste${items.length > 1 ? 's' : ''}`;
  if (!items.length) {
    recommendations.innerHTML = '<div class="recommendations-empty"><h3>Aucune formation correspondante trouvée.</h3><p>Aucune formation correspondant aux informations disponibles n’a été trouvée dans les sources référencées.</p></div>';
    return;
  }
  recommendations.innerHTML = `<div class="recommendation-list">${items.map((item, index) => {
    const pourquoiHtml = Array.isArray(item.pourquoi) ? item.pourquoi.map(echapper).join('<br>') : echapper(item.pourquoi || 'Profil compatible');
    const prerequisHtml = echapper(item.prerequis || 'Sélection sur dossier BACC');
    const debouchesHtml = echapper(item.debouches || 'Information non documentée');
    return `
    <article class="recommendation-card"><span class="recommendation-rank">0${index + 1}</span><h3>${echapper(item.parcours || '')}</h3>
      <div class="relevance"><span>Pertinence</span><div class="bar"><i style="width:${item.pertinence || 50}%"></i></div><span>${item.pertinence || 50}%</span></div>
      <div class="detail-grid"><div><strong>Pourquoi cette piste ?</strong><p>${pourquoiHtml}</p></div><div><strong>Prérequis</strong><p>${prerequisHtml}</p></div><div><strong>Débouchés</strong><p>${debouchesHtml}</p></div></div>
      <span class="uncertainty">Incertitude : ${echapper(item.incertitude || 'Modérée')}</span>
    </article>`;
  }).join('')}</div>`;
}

function afficherSources(sources = []) {
  sourcesList.innerHTML = sources.length ? sources.map(source => `<article class="source-item"><strong>${echapper(source.nom)}</strong><span>${echapper(source.type)} · ${echapper(source.origine)} · ${echapper(source.statut)}</span>${source.url ? `<a href="${encodeURI(source.url)}" target="_blank" rel="noreferrer">${echapper(source.url)}</a>` : '<span>URL non fournie par le backend.</span>'}<span>${echapper(source.date || 'Date non fournie par le backend.')}</span></article>`).join('') : '<p class="muted-text">Les sources retournées par le backend apparaîtront ici.</p>';
}

function afficherTrace(trace = {}) {
  const etapes = [['Question utilisateur', trace.question], ['Profil utilisé', trace.profil], ['Outils appelés', trace.outils], ['Résultats et sources', trace.resultats]];
  traceContent.innerHTML = etapes.map((etape, index) => `${index ? '<div class="trace-line"></div>' : ''}<div class="trace-step"><b>${etape[0]}</b><span>${echapper(etape[1] || 'Information non disponible')}</span></div>`).join('');
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  profilCourant = lireProfil();
  afficherChargement();
  afficherEtat('Analyse du profil en cours...');
  try {
    const resultat = await analyserProfil(profilCourant);
    afficherRecommandations(resultat);
    afficherSources(resultat.sources);
    afficherTrace(resultat.tracabilite);
    if (!resultat.erreur) afficherEtat('Voici les pistes qui semblent les plus pertinentes à partir des informations disponibles.');
  } catch (error) {
    resultCount.textContent = 'Service indisponible';
    recommendations.innerHTML = '<div class="recommendations-empty"><div class="empty-symbol" aria-hidden="true">!</div><h3>Le service ORIENT’IA est momentanément indisponible.</h3><p>Réessayez dans quelques instants.</p></div>';
    afficherEtat('Le service ORIENT’IA est momentanément indisponible.');
  }
});

chatForm.addEventListener('submit', async event => {
  event.preventDefault();
  const input = document.querySelector('#chat-input');
  const message = input.value.trim();
  if (!message) return;
  afficherEtat(message, 'user');
  input.value = '';
  const bouton = chatForm.querySelector('button');
  bouton.disabled = true;
  bouton.textContent = '...';
  try {
    const resultat = await envoyerMessage(message, profilCourant);
    afficherEtat(resultat.message || 'Information non disponible dans les sources référencées.');
    if (resultat.tracabilite) afficherTrace(resultat.tracabilite);
  } catch (error) {
    afficherEtat('Le service ORIENT’IA est momentanément indisponible.');
  } finally {
    bouton.disabled = false;
    bouton.innerHTML = 'Envoyer <span aria-hidden="true">↗</span>';
  }
});

if (!modeDemonstration) {
  demoNotice.hidden = true;
  connectionLabel.textContent = 'Backend connecté';
}
