const API_URL = "/risk/check";


const locationButton =
    document.getElementById("locationButton");

const loading =
    document.getElementById("loading");

const resultCard =
    document.getElementById("resultCard");

const outsideCard =
    document.getElementById("outsideCard");

const errorCard =
    document.getElementById("errorCard");

const statusText =
    document.getElementById("statusText");

const statusDot =
    document.getElementById("statusDot");


locationButton.addEventListener(
    "click",
    getUserLocation
);


function getUserLocation() {

    hideAllMessages();

    if (!navigator.geolocation) {

        showError(
            "Geolocation is not supported by this browser."
        );

        return;
    }


    showLoading();


    navigator.geolocation.getCurrentPosition(
        sendLocationToAPI,
        handleLocationError,
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        }
    );
}


async function sendLocationToAPI(position) {

    const latitude =
        position.coords.latitude;

    const longitude =
        position.coords.longitude;


    try {

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    latitude: latitude,
                    longitude: longitude
                })
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Risk assessment failed."
            );
        }


        hideLoading();


        if (data.inside_risk_zone) {

            showRiskResult(data);

        } else {

            showOutsideResult(data);
        }

    }

    catch (error) {

        hideLoading();

        showError(
            error.message
        );
    }
}


function showRiskResult(data) {

    resultCard.classList.remove("hidden");

    document.getElementById("state")
        .textContent = data.state || "-";

    document.getElementById("district")
        .textContent = data.district || "-";

    document.getElementById("latitude")
        .textContent =
        Number(data.latitude).toFixed(6);

    document.getElementById("longitude")
        .textContent =
        Number(data.longitude).toFixed(6);


    document.getElementById("riskShare")
        .textContent =
        Number(
            data.predicted_risk_share
        ).toFixed(4);


    const riskBand =
        document.getElementById("riskBand");


    riskBand.textContent =
        data.risk_band || "-";


    riskBand.className =
        "risk-band";


    if (data.risk_band === "LOW") {

        riskBand.classList.add(
            "risk-low"
        );

    }

    else if (data.risk_band === "MEDIUM") {

        riskBand.classList.add(
            "risk-medium"
        );

    }

    else if (data.risk_band === "HIGH") {

        riskBand.classList.add(
            "risk-high"
        );
    }


    document.getElementById(
        "resultMessage"
    ).textContent =
        data.message || "";


    setStatus(
        "Risk assessment complete",
        true
    );
}


function showOutsideResult(data) {

    outsideCard.classList.remove(
        "hidden"
    );


    document.getElementById(
        "outsideMessage"
    ).textContent =
        data.message ||
        "Location is outside the currently mapped risk zones.";


    setStatus(
        "Location not mapped",
        false
    );
}


function showLoading() {

    loading.classList.remove(
        "hidden"
    );

    locationButton.disabled = true;

    locationButton.textContent =
        "Checking Location...";

    setStatus(
        "Checking location",
        true
    );
}


function hideLoading() {

    loading.classList.add(
        "hidden"
    );

    locationButton.disabled = false;

    locationButton.textContent =
        "📍 Check My Location";
}


function hideAllMessages() {

    resultCard.classList.add(
        "hidden"
    );

    outsideCard.classList.add(
        "hidden"
    );

    errorCard.classList.add(
        "hidden"
    );
}


function showError(message) {

    errorCard.classList.remove(
        "hidden"
    );

    document.getElementById(
        "errorMessage"
    ).textContent = message;


    setStatus(
        "Error",
        false
    );
}


function handleLocationError(error) {

    hideLoading();

    let message =
        "Unable to retrieve your location.";


    if (error.code === 1) {

        message =
            "Location permission was denied. Please allow location access in your browser.";

    }

    else if (error.code === 2) {

        message =
            "Your location could not be determined.";

    }

    else if (error.code === 3) {

        message =
            "Location request timed out. Please try again.";
    }


    showError(message);
}


function setStatus(
    message,
    active
) {

    statusText.textContent =
        message;

    statusDot.style.background =
        active
            ? "#22c55e"
            : "#9ca3af";
}