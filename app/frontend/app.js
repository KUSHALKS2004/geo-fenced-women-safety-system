/* ============================================================
   API CONFIGURATION
   ============================================================ */

const API_URL = "/risk/check";


/*
============================================================
VERIFIED DEMO LOCATION

This is the Anuppur location that we already tested successfully
through the FastAPI Swagger endpoint.
============================================================
*/

const DEMO_LATITUDE = 23.035743284012398;

const DEMO_LONGITUDE = 81.39049233657985;


/* ============================================================
   HTML ELEMENTS
   ============================================================ */

const locationButton =
    document.getElementById("locationButton");


const demoButton =
    document.getElementById("demoButton");


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


const warningBox =
    document.getElementById("warningBox");


const warningTitle =
    document.getElementById("warningTitle");


const warningMessage =
    document.getElementById("warningMessage");


const emergencyBox =
    document.getElementById("emergencyBox");


const sosButton =
    document.getElementById("sosButton");


/* ============================================================
   BUTTON EVENTS
   ============================================================ */


/*
------------------------------------------------------------
REAL USER LOCATION
------------------------------------------------------------
*/

locationButton.addEventListener(
    "click",
    getUserLocation
);


/*
------------------------------------------------------------
DEMO LOCATION
------------------------------------------------------------
*/

demoButton.addEventListener(
    "click",
    runDemoLocation
);


/* ============================================================
   GET REAL USER LOCATION
   ============================================================ */

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


/* ============================================================
   DEMO LOCATION
   ============================================================ */

function runDemoLocation() {

    hideAllMessages();

    showLoading();

    sendDemoLocationToAPI();
}


/* ============================================================
   SEND REAL LOCATION TO BACKEND
   ============================================================ */

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

        }

        else {

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


/* ============================================================
   SEND DEMO LOCATION TO BACKEND
   ============================================================ */

async function sendDemoLocationToAPI() {

    try {

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    latitude: DEMO_LATITUDE,
                    longitude: DEMO_LONGITUDE
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

        }

        else {

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


/* ============================================================
   SHOW RISK RESULT
   ============================================================ */

function showRiskResult(data) {

    resultCard.classList.remove(
        "hidden"
    );


    /* --------------------------------------------------------
       LOCATION INFORMATION
       -------------------------------------------------------- */

    document.getElementById("state")
        .textContent =
        data.state || "-";


    document.getElementById("district")
        .textContent =
        data.district || "-";


    document.getElementById("latitude")
        .textContent =
        Number(data.latitude).toFixed(6);


    document.getElementById("longitude")
        .textContent =
        Number(data.longitude).toFixed(6);


    /* --------------------------------------------------------
       RISK SHARE
       -------------------------------------------------------- */

    document.getElementById("riskShare")
        .textContent =
        Number(
            data.predicted_risk_share
        ).toFixed(4);


    /* --------------------------------------------------------
       RISK BAND
       -------------------------------------------------------- */

    const riskBand =
        document.getElementById("riskBand");


    riskBand.textContent =
        data.risk_band || "-";


    riskBand.className =
        "risk-band";


    /* --------------------------------------------------------
       RESET WARNING AND EMERGENCY SECTIONS
       -------------------------------------------------------- */

    warningBox.classList.add(
        "hidden"
    );


    emergencyBox.classList.add(
        "hidden"
    );


    /* --------------------------------------------------------
       RISK CLASSIFICATION
       -------------------------------------------------------- */

    if (data.risk_band === "LOW") {

        riskBand.classList.add(
            "risk-low"
        );


        showLowRiskWarning();

    }


    else if (data.risk_band === "MEDIUM") {

        riskBand.classList.add(
            "risk-medium"
        );


        showMediumRiskWarning();

    }


    else if (data.risk_band === "HIGH") {

        riskBand.classList.add(
            "risk-high"
        );


        showHighRiskWarning();
    }


    /* --------------------------------------------------------
       BACKEND MESSAGE
       -------------------------------------------------------- */

    document.getElementById(
        "resultMessage"
    ).textContent =
        data.message || "";


    setStatus(
        "Risk assessment complete",
        true
    );
}


/* ============================================================
   LOW RISK WARNING
   ============================================================ */

function showLowRiskWarning() {

    warningBox.classList.remove(
        "hidden"
    );


    warningTitle.textContent =
        "🟢 Normal Safety Information";


    warningMessage.textContent =
        "The current location is classified as LOW based on the model-based historical crime estimate. Continue following normal personal safety practices.";
}


/* ============================================================
   MEDIUM RISK WARNING
   ============================================================ */

function showMediumRiskWarning() {

    warningBox.classList.remove(
        "hidden"
    );


    warningTitle.textContent =
        "🟡 Caution";


    warningMessage.textContent =
        "The current location is classified as MEDIUM based on the model-based historical crime estimate. Stay aware of your surroundings and consider travelling through populated or well-connected areas.";
}


/* ============================================================
   HIGH RISK WARNING
   ============================================================ */

function showHighRiskWarning() {

    warningBox.classList.remove(
        "hidden"
    );


    warningTitle.textContent =
        "🔴 Safety Warning";


    warningMessage.textContent =
        "The current location is classified as HIGH based on the model-based historical crime estimate. Stay alert, consider moving toward a populated or public location, and keep emergency assistance available.";


    emergencyBox.classList.remove(
        "hidden"
    );
}


/* ============================================================
   LOCATION NOT MAPPED
   ============================================================ */

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


/* ============================================================
   LOADING
   ============================================================ */

function showLoading() {

    loading.classList.remove(
        "hidden"
    );


    locationButton.disabled = true;

    demoButton.disabled = true;


    locationButton.textContent =
        "Checking Location...";


    demoButton.textContent =
        "Checking Demo Location...";


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

    demoButton.disabled = false;


    locationButton.textContent =
        "📍 Check My Location";


    demoButton.textContent =
        "🧪 Test HIGH-Risk Location";
}


/* ============================================================
   HIDE ALL MESSAGES
   ============================================================ */

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


    warningBox.classList.add(
        "hidden"
    );


    emergencyBox.classList.add(
        "hidden"
    );
}


/* ============================================================
   SHOW ERROR
   ============================================================ */

function showError(message) {

    errorCard.classList.remove(
        "hidden"
    );


    document.getElementById(
        "errorMessage"
    ).textContent =
        message;


    setStatus(
        "Error",
        false
    );
}


/* ============================================================
   HANDLE LOCATION ERROR
   ============================================================ */

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


/* ============================================================
   STATUS
   ============================================================ */

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


/* ============================================================
   SOS BUTTON
   ============================================================ */

sosButton.addEventListener(
    "click",
    function () {

        alert(
            "SOS functionality will be connected to the emergency assistance module."
        );

    }
);