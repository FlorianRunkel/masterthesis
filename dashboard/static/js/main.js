document.addEventListener("DOMContentLoaded", function() {
    // "Add Experience" Button Funktionalität
    document.getElementById("addExperience").addEventListener("click", function() {
        const experiencesContainer = document.getElementById("experiences");
        const newExperience = document.createElement("div");
        newExperience.classList.add("experience-entry");
        
        // Die Felder für die neue Erfahrung
        newExperience.innerHTML = `
            <input type="text" placeholder="Firma" required>
            <input type="text" placeholder="Position" required>
            <input type="date" placeholder="Start-Datum" required>
            <input type="date" placeholder="End-Datum">
            <button type="button" class="removeExperience">🗑️</button>
        `;

        experiencesContainer.appendChild(newExperience);
    });

    // Funktion, um ein "Remove"-Button zu behandeln
    document.getElementById("experiences").addEventListener("click", function(event) {
        if (event.target && event.target.classList.contains("removeExperience")) {
            const experienceEntry = event.target.closest(".experience-entry");
            if (experienceEntry) {
                experienceEntry.remove();
            }
        }
    });

    // Dynamisch weitere Ausbildungseinträge hinzufügen
    const addEducationButton = document.getElementById('addEducation');
    const educationFieldsContainer = document.getElementById('educationFields');

    addEducationButton.addEventListener('click', function () {
        const educationEntry = document.createElement('div');
        educationEntry.classList.add('education-entry');
        
        const schoolInput = document.createElement('input');
        schoolInput.setAttribute('type', 'text');
        schoolInput.setAttribute('placeholder', 'Schule');
        
        const degreeInput = document.createElement('input');
        degreeInput.setAttribute('type', 'text');
        degreeInput.setAttribute('placeholder', 'Abschluss');
        
        const removeButton = document.createElement('button');
        removeButton.classList.add('remove-education');
        removeButton.textContent = 'Delete';
        
        educationEntry.appendChild(schoolInput);
        educationEntry.appendChild(degreeInput);
        educationEntry.appendChild(removeButton);
        
        educationFieldsContainer.appendChild(educationEntry);

        removeButton.addEventListener('click', function () {
            educationFieldsContainer.removeChild(educationEntry);
        });
    });

    // Kontakt-Slide öffnen
    function openContactSlide() {
        document.getElementById('contactSlide').classList.add('show');
    }

    // Kontakt-Slide schließen
    document.getElementById('closeSlide').addEventListener('click', function() {
        document.getElementById('contactSlide').classList.remove('show');
    });

    // Optional: Automatisch die Kontakt-Slide nach einer bestimmten Zeit anzeigen (z.B. nach 5 Sekunden)
    setTimeout(openContactSlide, 5000); // 5 Sekunden warten, bevor die Slide erscheint
});

// Formulardaten absenden
document.getElementById('careerForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Sammeln der Formulardaten
    const formData = new FormData(event.target);
    const formObj = Object.fromEntries(formData);

    // Sammeln der Berufserfahrung
    const experiences = [];
    const experienceEntries = document.querySelectorAll('.experience-entry');
    experienceEntries.forEach(entry => {
        const company = entry.querySelector('input[placeholder="Firma"]').value;
        const position = entry.querySelector('input[placeholder="Position"]').value;
        const startDate = entry.querySelector('input[placeholder="Start-Datum"]').value;
        const endDate = entry.querySelector('input[placeholder="End-Datum"]').value;

        experiences.push({ company, position, startDate, endDate });
    });

    formObj.experiences = experiences;
    formObj.modelType = document.getElementById('modelType').value;

    try {
        // Senden der Daten an das Backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formObj)
        });

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        // Vorhersagebox anzeigen
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.style.display = 'block';

        const recommendations = Array.isArray(result.recommendations)
        ? result.recommendations
        : [result.recommendations];

        // Vorhersagetext einfügen
        const predictionElement = document.getElementById('prediction');
        predictionElement.innerHTML = `
            <div class="prediction-content">
                <h4>Nächster Karriereschritt</h4>                
                <h4>Konfidenz</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
                <p>${Math.round(result.confidence * 100)}%</p>
                
                <h4>Empfehlungen</h4>
                <ul>
                    ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;

    } catch (error) {
        console.error('Fehler bei der Vorhersage:', error);
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.style.display = 'block';
        const predictionElement = document.getElementById('prediction');
        predictionElement.innerHTML = `
            <div class="error-message">
                <p>Es ist ein Fehler aufgetreten: ${error.message}</p>
            </div>
        `;
    }
});
