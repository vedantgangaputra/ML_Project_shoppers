// This file handles form submission and API call to /predict endpoint

document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault(); // prevent default form submission

    // collect all form values into an object
    const data = {
        Administrative: parseInt(document.getElementById('Administrative').value),
        Administrative_Duration: parseFloat(document.getElementById('Administrative_Duration').value),
        Informational: parseInt(document.getElementById('Informational').value),
        Informational_Duration: parseFloat(document.getElementById('Informational_Duration').value),
        ProductRelated: parseInt(document.getElementById('ProductRelated').value),
        ProductRelated_Duration: parseFloat(document.getElementById('ProductRelated_Duration').value),
        BounceRates: parseFloat(document.getElementById('BounceRates').value),
        ExitRates: parseFloat(document.getElementById('ExitRates').value),
        PageValues: parseFloat(document.getElementById('PageValues').value),
        SpecialDay: parseFloat(document.getElementById('SpecialDay').value),
        Month: document.getElementById('Month').value,
        OperatingSystems: parseInt(document.getElementById('OperatingSystems').value),
        Browser: parseInt(document.getElementById('Browser').value),
        Region: parseInt(document.getElementById('Region').value),
        TrafficType: parseInt(document.getElementById('TrafficType').value),
        VisitorType: document.getElementById('VisitorType').value,
        Weekend: parseInt(document.getElementById('Weekend').value),
    };

    const resultDiv = document.getElementById('result'); // get result div
    resultDiv.className = 'result'; // reset classes
    resultDiv.textContent = 'Predicting...'; // show loading text
    resultDiv.classList.remove('hidden'); // show result div

    try {
        // call predict API endpoint
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json(); // parse response

        if (!response.ok) {
            const errorText = result?.detail || result?.message || result?.error || response.statusText || 'Unknown error';
            resultDiv.classList.add('error');
            resultDiv.innerHTML = `❌ Error: ${errorText}`;
            return;
        }

        const probabilityRaw = result?.probability;
        const probabilityValue = Number.isFinite(Number(probabilityRaw)) ? Number(probabilityRaw) : 0.0;
        const probabilityLabel = Number.isFinite(probabilityValue)
            ? `${(probabilityValue * 100).toFixed(1)}%`
            : 'N/A';

        if (result.prediction === 1) {
            resultDiv.classList.add('purchase'); // green style
            resultDiv.innerHTML = `✅ ${result.message} <br><small>Probability: ${probabilityLabel}</small>`;
        } else if (result.prediction === 0) {
            resultDiv.classList.add('no-purchase'); // red style
            resultDiv.innerHTML = `❌ ${result.message} <br><small>Probability: ${probabilityLabel}</small>`;
        } else {
            resultDiv.classList.add('error');
            resultDiv.innerHTML = `❌ Unexpected response from server.`;
        }

    } catch (error) {
        resultDiv.classList.add('error');
        resultDiv.innerHTML = '❌ Error: ' + (error.message || 'Network or parsing failure');
    }
});