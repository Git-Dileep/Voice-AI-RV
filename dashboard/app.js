const ACTION_API_URL = "http://127.0.0.1:8000/api/actions";
const ORCHESTRATOR_API_URL = "http://127.0.0.1:8001/api/chat";


// Send message to Agent Orchestrator
async function sendMessage() {

    const text = document.getElementById("userInput").value;
    const responseElement = document.getElementById("assistantResponse");

    if (!text) {
        responseElement.textContent = "Please enter a message.";
        return;
    }

    responseElement.textContent = "Processing...";

    try {

        const response = await fetch(ORCHESTRATOR_API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                lang: "en",
                confidence: 1.0
            })
        });

        const data = await response.json();

        if (data.ok) {
            responseElement.textContent = data.response;
        } else {
            responseElement.textContent = "Something went wrong.";
        }

    } catch (error) {

        responseElement.textContent =
            "Could not connect to Orchestrator API.";

        console.error(error);
    }
}


// Get weather
async function getWeather() {

    const location = document.getElementById("location").value;
    const result = document.getElementById("weatherResult");

    try {

        const response = await fetch(
            `${ACTION_API_URL}/get_weather`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    location: location
                })
            }
        );

        const data = await response.json();

        result.textContent =
            `${data.location}: ${data.weather}, ${data.temperature}`;

    } catch (error) {

        result.textContent =
            "Could not connect to Action API.";

        console.error(error);
    }
}


// Update record
async function updateRecord() {

    const recordId =
        document.getElementById("recordId").value;

    const value =
        document.getElementById("recordValue").value;

    const result =
        document.getElementById("updateResult");

    try {

        const response = await fetch(
            `${ACTION_API_URL}/update_record`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    record_id: recordId,
                    value: value
                })
            }
        );

        const data = await response.json();

        if (data.ok) {

            result.textContent =
                `Record ${data.record_id} updated successfully.`;

        } else {

            result.textContent =
                `Error: ${data.error}`;
        }

    } catch (error) {

        result.textContent =
            "Could not connect to Action API.";

        console.error(error);
    }
}