import { analyserProfil, envoyerMessage, modeDemonstration } from './api-client.js';

const form = document.querySelector('#profile-form');
const chatForm = document.querySelector('#chat-form');
const chatInput = document.querySelector('#chat-input');
const chatLog = document.querySelector('#chat-log');
const recommendations = document.querySelector('#recommendations');
const sourcesList = document.querySelector('#sources-list');
const resultCount = document.querySelector('#result-count');
const demoNotice = document.querySelector('#demo-notice');
const connectionLabel = document.querySelector('#connection-label');
const traceContent = document.querySelector('#trace-content');
const btnAnalyser = document.querySelector('#btn-analyser');

let profilCourant = {};

const valeursCochees = nom => [...document.querySelectorAll(`input[name="${nom}"]:checked`)].map(input => input.value);

const lireProfil = () => ({
  niveau: document.querySelector('#niveau').value,
  matieres_preferees: document.querySelector('#matieres').value,
  competences: document.querySelector('#competences').value,
  centres_interet: valeursCochees('centres_interet'),
  preferences_professionnelles: document.querySelector('#preference').value,
  environnement_travail: document.querySelector('#environnement').value
});

function echapper(value) {
  if (!value) return '';
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

function formaterMarkdown(texte) {
  if (!texte) return '';
  let formatte = echapper(texte);
  
  // Titres Markdown
  formatte = formatte.replace(/^### (.*$)/gim, '<strong style="display:block;margin:6px 0 3px;color:#1e293b;font-size:13.5px;">$1</strong>');
  formatte = formatte.replace(/^## (.*$)/gim, '<strong style="display:block;margin:8px 0 4px;color:#0f172a;font-size:14px;">$1</strong>');
  
  // Gras et italique
  formatte = formatte.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatte = formatte.replace(/\*(.*?)\*/g, '<em>$1</em>');
  
  // Puces
  formatte = formatte.replace(/^- (.*$)/gim, '<div style="margin:2px 0;padding-left:12px;position:relative;"><span style="position:absolute;left:0;color:#4f46e5;">•</span> $1</div>');
  
  // Sauts de ligne
  formatte = formatte.replace(/\n\n/g, '<div style="height:8px;"></div>');
  formatte = formatte.replace(/\n/g, '<br>');
  
  return formatte;
}

function afficherEtat(message, type = 'assistant') {
  const element = document.createElement('div');
  element.className = `chat-msg ${type === 'user' ? 'user' : 'assistant'}`;
  
  if (type === 'user') {
    element.innerHTML = `
      <div class="msg-avatar">V</div>
      <div class="msg-bubble">
        <p>${echapper(message)}</p>
      </div>
    `;
  } else {
    element.innerHTML = `
      <div class="msg-avatar">O</div>
      <div class="msg-bubble">
        <div class="msg-meta">ORIENT’IA <span>Assistant</span></div>
        <div>${formaterMarkdown(message)}</div>
      </div>
    `;
  }
  
  chatLog.appendChild(element);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function afficherChargement() {
  resultCount.textContent = 'Calcul en cours...';
  recommendations.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon" style="animation: spin 1s linear infinite;">⏳</div>
      <h3>Analyse du profil en cours...</h3>
      <p>Sélection des filières les plus pertinentes parmi les 16 mentions de l'ISPM.</p>
    </div>
  `;
}

function afficherRecommandations(resultat) {
  if (!resultat || resultat.erreur === 'profil_incomplet') {
    resultCount.textContent = 'Profil incomplet';
    recommendations.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <h3>Profil à compléter</h3>
        <p>Veuillez sélectionner au moins votre niveau d'étude ou un centre d'intérêt.</p>
      </div>
    `;
    return;
  }
  
  const items = resultat.recommandations || [];
  resultCount.textContent = `${items.length} filière${items.length > 1 ? 's' : ''} trouvée${items.length > 1 ? 's' : ''}`;
  
  if (!items.length) {
    recommendations.innerHTML = `
      <div class="empty-state">
        <h3>Aucune filière correspondante</h3>
        <p>Essayez de préciser vos matières préférées ou de cocher des centres d'intérêt.</p>
      </div>
    `;
    return;
  }
  
  recommendations.innerHTML = `
    <div class="recommendation-list">
      ${items.map((item, index) => {
        const pourquoiHtml = Array.isArray(item.pourquoi) 
          ? item.pourquoi.map(echapper).join(' · ') 
          : echapper(item.pourquoi || 'Profil en adéquation avec les objectifs pédagogiques');
        const prerequisHtml = echapper(item.prerequis || 'Baccalauréat et sélection sur dossier');
        const debouchesHtml = echapper(item.debouches || 'Postes spécialisés en entreprise et ingénierie');
        const score = item.pertinence || 75;
        const isTop = index === 0;

        return `
          <article class="rec-card ${isTop ? 'top-match' : ''}">
            <div class="rec-card-top">
              <div>
                <span class="rec-badge-rank">Option 0${index + 1}</span>
                <h3 class="rec-title">${echapper(item.parcours || '')}</h3>
              </div>
              <div class="match-pill">
                <span class="match-score">${score}%</span>
                <span class="match-label">Match</span>
              </div>
            </div>

            <div class="rec-reasons">
              <strong>Pourquoi cette piste ?</strong>
              <span>${pourquoiHtml}</span>
            </div>

            <div class="rec-details-grid">
              <div class="rec-detail-box">
                <span class="detail-label">Prérequis d'accès</span>
                <span class="detail-text">${prerequisHtml}</span>
              </div>
              <div class="rec-detail-box">
                <span class="detail-label">Métiers & Débouchés</span>
                <span class="detail-text">${debouchesHtml}</span>
              </div>
            </div>

            <div class="rec-card-footer">
              <span class="uncertainty-tag">Repère indicatif vérifié ISPM</span>
              <button type="button" class="btn-ask-about" data-parcours="${echapper(item.parcours || '')}">
                💬 Poser une question sur cette filière
              </button>
            </div>
          </article>
        `;
      }).join('')}
    </div>
  `;

  // Événements pour poser des questions rapides sur une carte
  recommendations.querySelectorAll('.btn-ask-about').forEach(btn => {
    btn.addEventListener('click', () => {
      const nomP = btn.getAttribute('data-parcours');
      envoyerQuestionDirecte(`Peux-tu me donner plus de détails sur la filière ${nomP} (débouchés, conditions et matières) ?`);
    });
  });
}

function afficherSources(sources = []) {
  if (!sources.length) {
    sourcesList.innerHTML = '<p style="color:var(--slate-500);font-size:12px;">Sources documentaires consultées en direct.</p>';
    return;
  }
  
  sourcesList.innerHTML = sources.map(s => `
    <div class="source-item">
      <div class="source-title">${echapper(s.nom || s.titre || 'Source institutionnelle')}</div>
      <div class="source-tag">${echapper(s.type || 'Fiche officielle')} · ${echapper(s.origine || 'ISPM')}</div>
      ${s.url ? `<a href="${encodeURI(s.url)}" target="_blank" rel="noreferrer" class="source-link">${echapper(s.url)}</a>` : ''}
    </div>
  `).join('');
}

function afficherTrace(trace = {}) {
  const etapes = [
    ['Question ou Intention', trace.question || 'Analyse de profil'],
    ['Profil appliqué', trace.profil || 'Profil candidat actif'],
    ['Composants sollicités', trace.outils || 'Modèle ExtraTrees + Moteur RAG ISPM'],
    ['Résultats & Justification', trace.resultats || 'Recommandation basée sur le corpus institutionnel']
  ];
  
  traceContent.innerHTML = etapes.map(e => `
    <div class="trace-item">
      <div class="trace-key">${e[0]}</div>
      <div class="trace-val">${echapper(e[1])}</div>
    </div>
  `).join('');
}

async function envoyerQuestionDirecte(message) {
  if (!message) return;
  chatInput.value = '';
  afficherEtat(message, 'user');
  
  const bouton = chatForm.querySelector('button');
  bouton.disabled = true;
  
  try {
    const resultat = await envoyerMessage(message, profilCourant);
    afficherEtat(resultat.message || 'Information traitée par l’assistant.');
    if (resultat.tracabilite) afficherTrace(resultat.tracabilite);
  } catch (error) {
    afficherEtat('Le service ORIENT’IA est temporairement indisponible.');
  } finally {
    bouton.disabled = false;
  }
}

// Formulaire de profil
form.addEventListener('submit', async event => {
  event.preventDefault();
  profilCourant = lireProfil();
  afficherChargement();
  if (btnAnalyser) btnAnalyser.disabled = true;
  
  try {
    const resultat = await analyserProfil(profilCourant);
    afficherRecommandations(resultat);
    afficherSources(resultat.sources);
    if (resultat.tracabilite) afficherTrace(resultat.tracabilite);
  } catch (error) {
    resultCount.textContent = 'Service indisponible';
    recommendations.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">❌</div>
        <h3>Connexion au serveur impossible</h3>
        <p>Vérifiez que le serveur ORIENT'IA est démarré sur http://127.0.0.1:8000/.</p>
      </div>
    `;
  } finally {
    if (btnAnalyser) btnAnalyser.disabled = false;
  }
});

// Chat Form
chatForm.addEventListener('submit', async event => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  await envoyerQuestionDirecte(message);
});

// Suggestions rapides
document.querySelectorAll('.prompt-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const msg = chip.getAttribute('data-msg');
    envoyerQuestionDirecte(msg);
  });
});

if (!modeDemonstration) {
  if (demoNotice) demoNotice.hidden = true;
  if (connectionLabel) connectionLabel.textContent = 'Connecté au moteur IA';
}
