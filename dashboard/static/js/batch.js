async function uploadAndPredict() {
    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];
    const loader = document.getElementById("loader");
    const loaderContainer = document.querySelector(".loader-container");
    const resultsDiv = document.getElementById("resultTableContainer");
    
    if (!file) {
        alert("Bitte wähle eine CSV-Datei aus.");
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
        
        const data = await response.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="error-message">Fehler: ${data.error}</div>`;
            resultsDiv.style.display = "block";
            return;
        }
        
        renderTable(data.results);
        resultsDiv.style.display = "block";
    } catch (error) {
        console.error("Fehler beim Hochladen oder Verarbeiten:", error);
        resultsDiv.innerHTML = `<div class="error-message">Es gab ein Problem bei der Vorhersage: ${error.message}</div>`;
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
                <h3>Zusammenfassung</h3>
                <p>Erfolgreich verarbeitet: ${successCount}</p>
                <p>Fehler: ${errorCount}</p>
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>LinkedIn</th>
                        <th>Wechselwahrscheinlichkeit des Kandidaten</th>
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
                    <td><a href="${linkedin}" target="_blank">${linkedin}</a></td>
                    <td>${result.error}</td>
                </tr>
            `;
        } else {
            const confidence = result.confidence ? result.confidence[0] * 100 : 0;

            let probabilityClass;
            if (confidence < 30) {
                probabilityClass = 'probability-low';
            } else if (confidence < 70) {
                probabilityClass = 'probability-medium';
            } else {
                probabilityClass = 'probability-high';
            }

            html += `
                <tr>
                    <td>${name}</td>
                    <td><a href="${linkedin}" target="_blank">${linkedin}</a></td>
                    <td>
                        <div class="probability-wrapper">
                            <span class="probability-value">${confidence.toFixed(0)}%</span>
                            <div class="probability-bar-container">
                                <div class="probability-bar ${probabilityClass}" style="width: ${confidence}%"></div>
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