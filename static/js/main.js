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
            <button type="button" class="removeExperience">🗑️</button> <!-- Mülleimer Emoji -->
        `;

        // Füge die neue Erfahrung hinzu
        experiencesContainer.appendChild(newExperience);
    });

    // Funktion, um ein "Remove"-Button zu behandeln
    document.getElementById("experiences").addEventListener("click", function(event) {
        if (event.target && event.target.classList.contains("removeExperience")) {
            // Entferne das übergeordnete .experience-entry
            const experienceEntry = event.target.closest(".experience-entry");
            if (experienceEntry) {
                experienceEntry.remove();
            }
        }
    });

    // Dynamisch weitere Ausbildungseinträge hinzufügen
    const addEducationButton = document.getElementById('addEducation');
    const educationFieldsContainer = document.getElementById('educationFields');

    // Funktion zum Hinzufügen eines neuen Ausbildungsfeldes
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
        
        // Anfügen der neuen Inputs und des Buttons zum Container
        educationEntry.appendChild(schoolInput);
        educationEntry.appendChild(degreeInput);
        educationEntry.appendChild(removeButton);
        
        educationFieldsContainer.appendChild(educationEntry);

        // Event Listener für den Entfernen-Button
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
    event.preventDefault(); // Verhindert das standardmäßige Absenden des Formulars

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

    // Auswahl des Modells
    formObj.modelType = document.getElementById('modelType').value; // Das Modell wird ausgelesen

    // Senden der Daten an das Backend
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formObj)
    });

    const result = await response.json();

    // Vorhersagetext einfügen
    const predictionText = result.prediction;
    const predictionElement = document.getElementById('prediction');
    predictionElement.textContent = predictionText;

    // Vorhersagebox anzeigen
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.style.display = 'block';  // Box sichtbar machen
});
