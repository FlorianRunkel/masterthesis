async function uploadAndPredict() {
    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];
    const loader = document.getElementById("loader");
    const loaderContainer = document.querySelector(".loader-container-prediction");
    const resultsDiv = document.getElementById("resultTableContainer");
    
    if (!file) {
        alert("Bitte wähle eine CSV-Datei aus.");
        return;
    }

    if (!file.name.endsWith('.csv')) {
        alert("Bitte wähle eine gültige CSV-Datei aus.");
        return;
    }

    if (!resultsDiv) {
        console.error("Fehler: #resultTableContainer nicht gefunden");
        return;
    }
    
    // Loader anzeigen
    loader.style.display = "block";
    loaderContainer.style.display = "flex";
    
    // Ergebnisse ausblenden
    resultsDiv.style.display = "none";
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const response = await fetch("/predict-batch", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `
                <div class="error-message">
                    <h4>Fehler bei der Verarbeitung</h4>
                    <p>${data.error}</p>
                    <p>Bitte überprüfe das Format deiner CSV-Datei.</p>
                </div>`;
            resultsDiv.style.display = "block";
            return;
        }
        
        renderTable(data.results);
        resultsDiv.style.display = "block";
    } catch (error) {
        console.error("Fehler beim Hochladen oder Verarbeiten:", error);
        resultsDiv.innerHTML = `
            <div class="error-message">
                <h4>Fehler bei der Verarbeitung</h4>
                <p>Es gab ein Problem bei der Vorhersage: ${error.message}</p>
                <p>Bitte stelle sicher, dass deine CSV-Datei folgende Spalten enthält:</p>
                <ul>
                    <li>firstName (Vorname)</li>
                    <li>lastName (Nachname)</li>
                    <li>linkedinProfile (LinkedIn-URL)</li>
                    <li>positions (Berufserfahrungen im JSON-Format)</li>
                </ul>
            </div>`;
        resultsDiv.style.display = "block";
    } finally {
        // Loader ausblenden
        loader.style.display = "none";
        loaderContainer.style.display = "none";
    }
}

function renderTable(results) {
    const resultsDiv = document.getElementById("resultTableContainer");

    if (!resultsDiv) {
        console.error("Fehler: #resultTableContainer nicht gefunden");
        return;
    }

    const successCount = results.filter(r => !r.error).length;
    const errorCount = results.filter(r => r.error).length;

    let html = `
        <div class="table-container">
            <div class="summary">
                <h3>Zusammenfassung der Batch-Verarbeitung</h3>
                <p><strong>Erfolgreich verarbeitet:</strong> ${successCount} Kandidaten</p>
                <p><strong>Fehler:</strong> ${errorCount} Kandidaten</p>
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>LinkedIn</th>
                        <th>Wechselwahrscheinlichkeit</th>
                    </tr>
                </thead>
                <tbody>
    `;

    results.forEach(result => {
        const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Nicht angegeben';
        const linkedin = result.linkedinProfile || 'Nicht angegeben';

        if (result.error) {
            html += `
                <tr class="error-row">
                    <td>${name}</td>
                    <td><a href="${linkedin}" target="_blank" rel="noopener noreferrer" style="color: #666; text-decoration: none;">${linkedin}</a></td>
                    <td colspan="2">${result.error}</td>
                </tr>
            `;
        } else {
            const confidence = result.confidence ? result.confidence[0] * 100 : 0;
            const recommendations = result.recommendations || [];
            const features = result.features || {};

            let probabilityClass;
            let probabilityText;
            if (confidence < 50) {
                probabilityClass = 'probability-low';
                probabilityText = 'Niedrig';
            } else if (confidence < 75) {
                probabilityClass = 'probability-medium';
                probabilityText = 'Mittel';
            } else {
                probabilityClass = 'probability-high';
                probabilityText = 'Hoch';
            }

            html += `
                <tr>
                    <td>${name}</td>
                    <td><a href="${linkedin}" target="_blank" rel="noopener noreferrer" style="color: #333; text-decoration: none;">${linkedin}</a></td>
                    <td>
                        <div class="probability-wrapper">
                            <span class="probability-value ${probabilityClass}-text">${confidence.toFixed(0)}%</span>
                            <div class="probability-bar-container">
                                <div class="probability-bar ${probabilityClass}" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <button class="details-btn" onclick="toggleDetails(this)" data-recommendations='${JSON.stringify(recommendations)}' data-features='${JSON.stringify(features)}'>
                            Details anzeigen
                        </button>
                    </td>
                </tr>
                <tr class="details-row" style="display: none;">
                    <td colspan="4">
                        <div class="recommendations-container">
                            <div class="features-section">
                                <h4>Verwendete Features:</h4>
                                <ul>
                                    ${Object.entries(features).map(([key, value]) => 
                                        `<li><strong>${key}:</strong> ${value.toFixed(3)}</li>`
                                    ).join('')}
                                </ul>
                            </div>
                            <div class="recommendations-section">
                                <h4>Empfehlungen:</h4>
                                <ul>
                                    ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        }
    });

    html += `
                </tbody>
            </table>
        </div>
    `;

    resultsDiv.innerHTML = html;
}

function toggleDetails(button) {
    const detailsRow = button.closest('tr').nextElementSibling;
    const isHidden = detailsRow.style.display === 'none';
    
    detailsRow.style.display = isHidden ? 'table-row' : 'none';
    button.textContent = isHidden ? 'Details ausblenden' : 'Details anzeigen';
}

// Dateiname anzeigen wenn eine Datei ausgewählt wurde
document.getElementById('csvFile').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Keine Datei ausgewählt';
    document.getElementById('file-name').textContent = fileName;
});