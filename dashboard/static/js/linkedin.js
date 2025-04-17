async function analyzeProfil() {
    const linkedinUrl = document.getElementById('linkedinUrl');
    const loaderContainer = document.getElementById('loaderContainer');
    const loaderStatus = document.getElementById('loaderStatus');
    const profileContainer = document.getElementById('profileContainer');
    const resultContainer = document.getElementById('resultTableContainer');

    if (!linkedinUrl || !loaderContainer || !loaderStatus || !profileContainer || !resultContainer) {
        console.error('Erforderliche DOM-Elemente nicht gefunden');
        alert('Ein technischer Fehler ist aufgetreten. Bitte laden Sie die Seite neu.');
        return;
    }

    // Validiere URL
    if (!linkedinUrl.value || !linkedinUrl.value.includes('linkedin.com/in/')) {
        showError('Bitte geben Sie eine gültige LinkedIn-Profil-URL ein (z.B. https://www.linkedin.com/in/username)');
        return;
    }

    // Reset Container
    profileContainer.style.display = 'none';
    resultContainer.style.display = 'none';
    loaderContainer.style.display = 'flex';
    
    try {
        // 1. Scraping des LinkedIn-Profils
        loaderStatus.textContent = 'LinkedIn-Profil wird geladen...';
        const profileResponse = await fetch('/scrape-linkedin', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: linkedinUrl.value })
        });

        if (!profileResponse.ok) {
            const errorData = await profileResponse.json();
            throw new Error(errorData.error || 'Fehler beim Laden des LinkedIn-Profils');
        }

        const profileData = await profileResponse.json();
        
        // 2. Anzeigen des Profils
        displayProfile(profileData);
        profileContainer.style.display = 'block';

        // 3. Karriere-Vorhersage
        loaderStatus.textContent = 'Karriere-Analyse wird durchgeführt...';
        
        // Formatiere die Erfahrungen für die Analyse
        const careerHistory = profileData.experience.map(exp => ({
            position: exp.title,
            company: exp.company,
            startDate: exp.duration.split(' - ')[0],
            endDate: exp.duration.split(' - ')[1] === 'Present' ? null : exp.duration.split(' - ')[1]
        }));

        const analysisResponse = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                experiences: careerHistory,
                modelType: 'tft'
            })
        });

        if (!analysisResponse.ok) {
            const errorData = await analysisResponse.json();
            throw new Error(errorData.error || 'Fehler bei der Karriere-Analyse');
        }

        const analysisData = await analysisResponse.json();
        
        // 4. Anzeigen der Ergebnisse
        displayResults(analysisData);
        resultContainer.style.display = 'block';

        // 5. Verstecke Loader
        loaderContainer.style.display = 'none';

    } catch (error) {
        console.error('Fehler:', error);
        showError(error.message);
        loaderContainer.style.display = 'none';
    }
}

function displayProfile(profileData) {
    const profileSection = document.getElementById('profile-section');
    if (!profileSection) {
        console.error('Profile section nicht gefunden');
        return;
    }
    profileSection.innerHTML = '';

    // Profilbild und Hauptinformationen
    const profileHeader = document.createElement('div');
    profileHeader.className = 'profile-header';
    
    // Profilbild
    if (profileData.imageUrl) {
        const profileImage = document.createElement('img');
        profileImage.src = profileData.imageUrl;
        profileImage.alt = 'Profilbild';
        profileImage.className = 'profile-image';
        profileHeader.appendChild(profileImage);
    }

    // Profilinformationen
    const profileInfo = document.createElement('div');
    profileInfo.className = 'profile-info';
    
    const name = document.createElement('h2');
    name.textContent = profileData.name || 'Kein Name verfügbar';
    profileInfo.appendChild(name);

    const title = document.createElement('p');
    title.className = 'profile-title';
    title.textContent = profileData.currentTitle || 'Keine Position angegeben';
    profileInfo.appendChild(title);

    const location = document.createElement('p');
    location.className = 'profile-location';
    location.textContent = profileData.location || 'Kein Standort angegeben';
    profileInfo.appendChild(location);

    if (profileData.industry) {
        const industry = document.createElement('p');
        industry.className = 'profile-industry';
        industry.textContent = `Branche: ${profileData.industry}`;
        profileInfo.appendChild(industry);
    }

    profileHeader.appendChild(profileInfo);
    profileSection.appendChild(profileHeader);

    // Zusammenfassung
    if (profileData.summary) {
        const summarySection = document.createElement('div');
        summarySection.className = 'profile-summary';
        const summaryTitle = document.createElement('h3');
        summaryTitle.textContent = 'Zusammenfassung';
        summarySection.appendChild(summaryTitle);
        const summaryText = document.createElement('p');
        summaryText.textContent = profileData.summary;
        summarySection.appendChild(summaryText);
        profileSection.appendChild(summarySection);
    }

    // Berufserfahrung
    if (profileData.experience && profileData.experience.length > 0) {
        const experienceSection = document.createElement('div');
        experienceSection.className = 'profile-experience';
        const experienceTitle = document.createElement('h3');
        experienceTitle.textContent = 'Berufserfahrung';
        experienceSection.appendChild(experienceTitle);

        profileData.experience.forEach(exp => {
            const expEntry = document.createElement('div');
            expEntry.className = 'experience-entry';

            const expHeader = document.createElement('div');
            expHeader.className = 'entry-header';

            const expTitle = document.createElement('h4');
            expTitle.textContent = exp.title || 'Keine Position angegeben';
            expHeader.appendChild(expTitle);

            const expCompany = document.createElement('p');
            expCompany.className = 'company';
            expCompany.textContent = exp.company || 'Kein Unternehmen angegeben';
            expHeader.appendChild(expCompany);

            const expDuration = document.createElement('p');
            expDuration.className = 'duration';
            expDuration.textContent = exp.duration || 'Kein Zeitraum angegeben';
            expHeader.appendChild(expDuration);

            expEntry.appendChild(expHeader);
            experienceSection.appendChild(expEntry);
        });
        profileSection.appendChild(experienceSection);
    }
}

function displayResults(data) {
    const resultContainer = document.getElementById('resultTableContainer');
    if (!resultContainer) {
        console.error('Result container nicht gefunden');
        return;
    }

    // Sicherstellen, dass recommendations ein Array ist
    const recommendations = Array.isArray(data.recommendations) 
        ? data.recommendations 
        : [data.recommendations];

    resultContainer.style.display = 'block';
    resultContainer.innerHTML = `
        <div class="prediction-section">
            <div class="probability-section">
                <div class="probability-header">
                    <h4>Wahrscheinlichkeit der Vorhersage</h4>
                    <span class="probability-value">${Math.round(data.confidence * 100)}%</span>
                </div>
                <div class="probability-bar-container">
                    <div class="probability-bar-single" id="probability-bar" style="width: ${data.confidence * 100}%"></div>
                </div>
            </div>
            <div class="recommendations-section">
                <h4>Empfohlene nächste Positionen</h4>
                <ul class="recommendation-list">
                    ${recommendations && recommendations.length > 0 
                        ? recommendations.map(rec => `<li class="recommendation-item">${rec}</li>`).join('')
                        : '<li class="recommendation-item">Keine Empfehlungen verfügbar</li>'
                    }
                </ul>
            </div>
        </div>
    `;

    // Setze die Farbe des Wahrscheinlichkeitsbalkens
    const probabilityBar = document.getElementById('probability-bar');
    if (probabilityBar) {
        const confidence = data.confidence * 100;
        if (confidence < 50) {
            probabilityBar.className = 'probability-bar-single probability-low-single';
        } else if (confidence < 75) {
            probabilityBar.className = 'probability-bar-single probability-medium-single';
        } else {
            probabilityBar.className = 'probability-bar-single probability-high-single';
        }
    }
}

function getProbabilityClass(probability) {
    if (probability < 0.4) return 'probability-low';
    if (probability < 0.7) return 'probability-medium';
    return 'probability-high';
}

function toggleDetails(button) {
    const detailsRow = button.closest('tr').nextElementSibling;
    const isHidden = detailsRow.style.display === 'none';
    detailsRow.style.display = isHidden ? 'table-row' : 'none';
    button.textContent = isHidden ? 'Verbergen' : 'Details';
}

function showError(message) {
    const resultContainer = document.getElementById('resultTableContainer');
    if (!resultContainer) {
        console.error('Result container nicht gefunden');
        alert('Ein Fehler ist aufgetreten: ' + message);
        return;
    }

    resultContainer.style.display = 'block';
    resultContainer.innerHTML = `
        <div class="error-message">
            <h4>Fehler</h4>
            <p>${message}</p>
            <ul>
                <li>Stellen Sie sicher, dass die URL korrekt ist</li>
                <li>Das Profil muss öffentlich zugänglich sein</li>
                <li>Versuchen Sie es später erneut</li>
            </ul>
        </div>
    `;
} 