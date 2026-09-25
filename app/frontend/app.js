/* =========================================================
   SAFEHER-AI FRONTEND
   Authentication + Risk Assessment + Live GPS + SOS
========================================================= */

const API_BASE = "";

function getAuthHeaders() {
    const token = localStorage.getItem("safeher_token");

    if (!token) {
        return {
            "Content-Type": "application/json"
        };
    }

    return {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
    };
}

const DEMO_LATITUDE = 23.035743284012398;
const DEMO_LONGITUDE = 81.39049233657985;


/* =========================================================
   AUTHENTICATION
========================================================= */

let currentUser = null;


/* =========================================================
   DOM HELPER
========================================================= */

function $(id) {
    return document.getElementById(id);
}


/* =========================================================
   PAGE INITIALIZATION
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    initializeAuthentication();

});


/* =========================================================
   AUTH INITIALIZATION
========================================================= */

function initializeAuthentication() {

    const savedUser =
        localStorage.getItem("safeher_user");

    const savedToken =
        localStorage.getItem("safeher_token");

    const demoMode =
        localStorage.getItem("safeher_demo_mode");


    if (savedUser && savedToken) {

        try {

            currentUser =
                JSON.parse(savedUser);

            showApplication();

            updateUserInterface();

        } catch (error) {

            console.log(
                "Saved login session could not be restored."
            );

            logoutUser(false);
        }

        return;
    }


    if (demoMode === "true") {

        currentUser = {
            full_name: "Demo User",
            email: "demo@safeher-ai.local",
            demo: true
        };

        showApplication();

        updateUserInterface();

        return;
    }


    showAuthentication();
}


/* =========================================================
   SHOW LOGIN
========================================================= */

function showLoginForm() {

    const loginContainer =
        $("loginFormContainer");

    const registerContainer =
        $("registerFormContainer");

    const loginTab =
        $("loginTab");

    const registerTab =
        $("registerTab");


    if (loginContainer) {

        loginContainer.style.display =
            "block";
    }


    if (registerContainer) {

        registerContainer.style.display =
            "none";
    }


    if (loginTab) {

        loginTab.classList.add("active");
    }


    if (registerTab) {

        registerTab.classList.remove("active");
    }


    clearAuthMessage();
}


/* =========================================================
   SHOW REGISTER
========================================================= */

function showRegisterForm() {

    const loginContainer =
        $("loginFormContainer");

    const registerContainer =
        $("registerFormContainer");

    const loginTab =
        $("loginTab");

    const registerTab =
        $("registerTab");


    if (loginContainer) {

        loginContainer.style.display =
            "none";
    }


    if (registerContainer) {

        registerContainer.style.display =
            "block";
    }


    if (loginTab) {

        loginTab.classList.remove("active");
    }


    if (registerTab) {

        registerTab.classList.add("active");
    }


    clearAuthMessage();
}


/* =========================================================
   AUTH MESSAGE
========================================================= */

function showAuthMessage(
    message,
    type = "error"
) {

    const box =
        $("authMessage");


    if (!box) {
        return;
    }


    box.textContent =
        message;

    box.className =
        `auth-message ${type}`;

    box.style.display =
        "block";
}


function clearAuthMessage() {

    const box =
        $("authMessage");


    if (!box) {
        return;
    }


    box.textContent =
        "";

    box.style.display =
        "none";

    box.className =
        "auth-message";
}


/* =========================================================
   REGISTER
========================================================= */

async function registerUser(event) {

    event.preventDefault();

    const name =
        $("registerName").value.trim();

    const email =
        $("registerEmail").value.trim();

    const phone =
        $("registerPhone").value.trim();

    const password =
        $("registerPassword").value;

    const confirmPassword =
        $("registerConfirmPassword").value;


    clearAuthMessage();


    if (!name) {

        showAuthMessage(
            "Please enter your full name."
        );

        return;
    }


    if (!email) {

        showAuthMessage(
            "Please enter your email address."
        );

        return;
    }


    if (password.length < 6) {

        showAuthMessage(
            "Password must contain at least 6 characters."
        );

        return;
    }


    if (password !== confirmPassword) {

        showAuthMessage(
            "Passwords do not match."
        );

        return;
    }


    const button =
        $("registerButton");


    if (button) {

        button.disabled = true;

        button.textContent =
            "Creating Account...";
    }


    try {

        const response =
            await fetch(
                `${API_BASE}/auth/register`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        full_name:
                            name,

                        email:
                            email,

                        phone:
                            phone || null,

                        password:
                            password
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Registration failed."
            );
        }


        showAuthMessage(
            "✓ Registration successful! Redirecting to login...",
            "success"
        );


        $("registerForm").reset();


        setTimeout(() => {

            showLoginForm();

            $("loginEmail").value =
                email;

            $("loginPassword").focus();

            showAuthMessage(
                "✓ Registration successful. Please login with your password.",
                "success"
            );

        }, 1500);


    } catch (error) {

        showAuthMessage(
            error.message ||
            "Unable to create account."
        );

    } finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "✨ Create Account";
        }
    }
}


/* =========================================================
   LOGIN
========================================================= */

async function loginUser(event) {

    event.preventDefault();


    const email =
        $("loginEmail").value.trim();

    const password =
        $("loginPassword").value;


    clearAuthMessage();


    if (!email || !password) {

        showAuthMessage(
            "Please enter your email and password."
        );

        return;
    }


    const button =
        $("loginButton");


    if (button) {

        button.disabled =
            true;

        button.textContent =
            "Logging in...";
    }


    try {

        const response =
            await fetch(
                `${API_BASE}/auth/login`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        email:
                            email,

                        password:
                            password
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Invalid email or password."
            );
        }


        localStorage.setItem(
            "safeher_token",
            data.access_token
        );


        localStorage.setItem(
            "safeher_user",
            JSON.stringify(data.user)
        );


        localStorage.removeItem(
            "safeher_demo_mode"
        );


        currentUser =
            data.user;


        showApplication();

        updateUserInterface();


    } catch (error) {

        showAuthMessage(
            error.message ||
            "Login failed."
        );

    } finally {

        if (button) {

            button.disabled =
                false;

            button.textContent =
                "🔐 Login";
        }
    }
}


/* =========================================================
   DEMO MODE
========================================================= */

function continueAsDemo() {

    localStorage.removeItem(
        "safeher_token"
    );

    localStorage.removeItem(
        "safeher_user"
    );


    localStorage.setItem(
        "safeher_demo_mode",
        "true"
    );


    currentUser = {

        full_name:
            "Demo User",

        email:
            "demo@safeher-ai.local",

        demo:
            true
    };


    showApplication();

    updateUserInterface();
}


/* =========================================================
   SHOW APPLICATION
========================================================= */

function showApplication() {

    const authScreen =
        $("authScreen");

    const appShell =
        $("appShell");


    if (authScreen) {

        authScreen.style.display =
            "none";
    }


    if (appShell) {

        appShell.style.display =
            "block";
    }
}


/* =========================================================
   SHOW AUTHENTICATION
========================================================= */

function showAuthentication() {

    const authScreen =
        $("authScreen");

    const appShell =
        $("appShell");


    if (authScreen) {

        authScreen.style.display =
            "flex";
    }


    if (appShell) {

        appShell.style.display =
            "none";
    }
}


/* =========================================================
   UPDATE USER UI
========================================================= */

function updateUserInterface() {

    const userName =
        $("userName");


    if (!userName) {
        return;
    }


    if (
        currentUser &&
        currentUser.full_name
    ) {

        userName.textContent =
            currentUser.full_name;

    } else {

        userName.textContent =
            "User";
    }
}


/* =========================================================
   LOGOUT
========================================================= */

function logoutUser(
    showMessage = true
) {

    stopLocationTracking();


    localStorage.removeItem(
        "safeher_token"
    );

    localStorage.removeItem(
        "safeher_user"
    );

    localStorage.removeItem(
        "safeher_demo_mode"
    );


    currentUser =
        null;


    showAuthentication();


    if (showMessage) {

        showLoginForm();

        showAuthMessage(
            "You have been logged out.",
            "success"
        );
    }
}


/* =========================================================
   DOM HELPERS FOR RESULTS
========================================================= */

function hideAllResults() {

    const result =
        $("result");

    const outsideCard =
        $("outsideCard");

    const errorCard =
        $("errorCard");

    const warningBox =
        $("warningBox");

    const emergencyBox =
        $("emergencyBox");


    if (result) {

        result.classList.add("hidden");

        result.style.display =
            "none";
    }


    if (outsideCard) {

        outsideCard.classList.add("hidden");

        outsideCard.style.display =
            "none";
    }


    if (errorCard) {

        errorCard.classList.add("hidden");

        errorCard.style.display =
            "none";
    }


    if (warningBox) {

        warningBox.classList.add("hidden");

        warningBox.style.display =
            "none";
    }


    if (emergencyBox) {

        emergencyBox.classList.add("hidden");

        emergencyBox.style.display =
            "none";
    }
}


function showLoading(show) {

    const loading =
        $("loading");


    if (!loading) {
        return;
    }


    if (show) {

        loading.classList.remove(
            "hidden"
        );

        loading.style.display =
            "block";

    } else {

        loading.classList.add(
            "hidden"
        );

        loading.style.display =
            "none";
    }
}


/* =========================================================
   LIVE LOCATION TRACKING
========================================================= */

let locationWatchId = null;
let lastRiskCheckTime = 0;


/* =========================================================
   STOP LOCATION TRACKING
========================================================= */

function stopLocationTracking() {

    if (locationWatchId !== null) {

        navigator.geolocation.clearWatch(
            locationWatchId
        );

        locationWatchId = null;

        console.log(
            "SafeHer-AI location tracking stopped."
        );
    }
}


/* =========================================================
   CHECK MY LOCATION
========================================================= */

function checkMyLocation() {

    hideAllResults();
    showLoading(true);

    console.log("=================================");
    console.log("SafeHer-AI GPS TRACKING STARTED");
    console.log("=================================");

    if (!navigator.geolocation) {

        console.error(
            "Geolocation is NOT supported by this browser."
        );

        showError(
            "Your browser does not support GPS location detection."
        );

        return;
    }

    /*
     * Stop previous GPS watcher
     */
    if (locationWatchId !== null) {

        navigator.geolocation.clearWatch(
            locationWatchId
        );

        locationWatchId = null;
    }

    /*
     * Reset request timer
     */
    lastRiskCheckTime = 0;

    /*
     * Start continuous GPS tracking
     */
    locationWatchId =
        navigator.geolocation.watchPosition(

            function(position) {

                console.log(
                    "================================="
                );

                console.log(
                    "GPS LOCATION RECEIVED"
                );

                console.log(
                    "Latitude:",
                    position.coords.latitude
                );

                console.log(
                    "Longitude:",
                    position.coords.longitude
                );

                console.log(
                    "Accuracy:",
                    position.coords.accuracy,
                    "meters"
                );

                console.log(
                    "================================="
                );


                const latitude =
                    position.coords.latitude;

                const longitude =
                    position.coords.longitude;


                /*
                 * Show coordinates immediately
                 * so we know GPS is working.
                 */

                if ($("latitudeValue")) {

                    $("latitudeValue").textContent =
                        latitude.toFixed(6);
                }


                if ($("longitudeValue")) {

                    $("longitudeValue").textContent =
                        longitude.toFixed(6);
                }


                /*
                 * Prevent excessive API requests.
                 */

                const now =
                    Date.now();


                if (
                    now - lastRiskCheckTime < 10000
                ) {

                    console.log(
                        "GPS updated, waiting before next risk request."
                    );

                    return;
                }


                lastRiskCheckTime =
                    now;


                console.log(
                    "Sending GPS location to /risk/check..."
                );


                checkRisk(
                    latitude,
                    longitude
                );

            },


            function(error) {

                console.error(
                    "================================="
                );

                console.error(
                    "GPS ERROR"
                );

                console.error(
                    "Error code:",
                    error.code
                );

                console.error(
                    "Error message:",
                    error.message
                );

                console.error(
                    "================================="
                );


                showLoading(false);


                let message =
                    "Unable to detect your current location.";


                if (
                    error.code ===
                    error.PERMISSION_DENIED
                ) {

                    message =
                        "Location permission was denied. Please allow location access for this website.";

                }

                else if (
                    error.code ===
                    error.POSITION_UNAVAILABLE
                ) {

                    message =
                        "Your device could not determine the current GPS location. Check Windows Location Services and try again.";

                }

                else if (
                    error.code ===
                    error.TIMEOUT
                ) {

                    message =
                        "GPS location detection timed out. Please try again outdoors or near a window.";

                }


                showError(
                    message
                );

            },


            {
                enableHighAccuracy: true,

                timeout: 30000,

                maximumAge: 0
            }
        );


    console.log(
        "GPS watcher started. Watch ID:",
        locationWatchId
    );
}


/* =========================================================
   DEMO LOCATION
========================================================= */

function runDemoLocation() {

    checkRisk(
        DEMO_LATITUDE,
        DEMO_LONGITUDE
    );
}


/* =========================================================
   RISK API
========================================================= */

async function checkRisk(
    latitude,
    longitude
) {

    hideAllResults();

    showLoading(true);


    try {

        const response =
            await fetch(
                `${API_BASE}/risk/check`,
                {
                    method: "POST",

                    headers: getAuthHeaders(),

                    body: JSON.stringify({

                        latitude:
                            latitude,

                        longitude:
                            longitude
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            /*
             * Authentication expired or
             * token is invalid.
             */

            if (
                response.status === 401
            ) {

                localStorage.removeItem(
                    "safeher_token"
                );

                localStorage.removeItem(
                    "safeher_user"
                );

                currentUser =
                    null;

                stopLocationTracking();

                showAuthentication();

                showLoginForm();

                showAuthMessage(
                    "Your login session has expired. Please login again."
                );

                return;
            }


            throw new Error(
                data.detail ||
                "Risk assessment failed."
            );
        }


        showLoading(false);


        if (
            data.inside_risk_zone
        ) {

            displayRiskResult(
                data
            );

        } else {

            displayOutsideCoverage(
                data
            );
        }


    } catch (error) {

        showLoading(false);

        showError(
            error.message ||
            "Unable to complete risk assessment."
        );
    }
}


/* =========================================================
   DISPLAY RISK
========================================================= */

function displayRiskResult(data) {

    const result =
        $("result");


    if (!result) {
        return;
    }


    result.classList.remove(
        "hidden"
    );

    result.style.display =
        "block";


    /* Location */

    $("stateValue").textContent =
        data.state ||
        "Unknown";


    $("districtValue").textContent =
        data.district ||
        "Unknown";


    $("latitudeValue").textContent =
        Number(
            data.latitude
        ).toFixed(6);


    $("longitudeValue").textContent =
        Number(
            data.longitude
        ).toFixed(6);


    /* Risk */

    const share =
        Number(
            data.predicted_risk_share ||
            0
        );


    const band =
        String(
            data.risk_band ||
            "UNKNOWN"
        ).toUpperCase();


    $("riskBand").textContent =
        band;


    $("riskValue").textContent =
        share.toFixed(4);


    /* Time */

    $("riskTime").textContent =
        "Assessed " +
        new Date()
            .toLocaleTimeString();


    /* Gauge */

    updateGauge(
        share,
        band
    );


    /* Message */

    updateRiskMessage(
        band,
        data.message
    );


    /* Warning + Emergency */

    updateSafetyUI(
        band
    );


    /* Zone */

    $("zoneText").textContent =
        "Location successfully matched to a SafeHer-AI risk zone.";
}


/* =========================================================
   GAUGE
========================================================= */

function updateGauge(
    share,
    band
) {

    const progress =
        $("gaugeProgress");


    if (!progress) {
        return;
    }


    const radius =
        86;


    const circumference =
        2 *
        Math.PI *
        radius;


    progress.style.strokeDasharray =
        circumference;


    /*
     * The predicted women-crime share
     * is not a probability.
     *
     * The gauge is only a visual
     * representation.
     */

    const normalized =
        Math.min(
            Math.max(
                share,
                0
            ),
            1
        );


    const offset =
        circumference -
        normalized *
        circumference;


    progress.style.strokeDashoffset =
        offset;


    let color =
        "#8b5cf6";


    if (
        band === "LOW"
    ) {

        color =
            "#22c55e";
    }


    if (
        band === "MEDIUM"
    ) {

        color =
            "#f59e0b";
    }


    if (
        band === "HIGH"
    ) {

        color =
            "#ef4444";
    }


    progress.style.stroke =
        color;


    if ($("riskBand")) {

        $("riskBand").style.color =
            color;
    }
}


/* =========================================================
   RISK MESSAGE
========================================================= */

function updateRiskMessage(
    band,
    apiMessage
) {

    let message =
        apiMessage ||
        "Model-based risk assessment completed.";


    if (
        band === "LOW"
    ) {

        message =
            "The current model-based estimate is in the LOW band. Continue normal safety practices.";


        $("riskIcon").textContent =
            "🟢";
    }


    if (
        band === "MEDIUM"
    ) {

        message =
            "The current model-based estimate is in the MEDIUM band. Consider additional caution.";


        $("riskIcon").textContent =
            "🟡";
    }


    if (
        band === "HIGH"
    ) {

        message =
            "The current model-based estimate is in the HIGH band. Exercise additional caution and consider the emergency assistance options.";


        $("riskIcon").textContent =
            "🔴";
    }


    $("riskMessage").textContent =
        message;
}


/* =========================================================
   SAFETY UI
========================================================= */

function updateSafetyUI(
    band
) {

    const warningBox =
        $("warningBox");

    const emergencyBox =
        $("emergencyBox");


    if (warningBox) {

        warningBox.classList.add(
            "hidden"
        );

        warningBox.style.display =
            "none";
    }


    if (emergencyBox) {

        emergencyBox.classList.add(
            "hidden"
        );

        emergencyBox.style.display =
            "none";
    }


    if (
        band === "MEDIUM"
    ) {

        if (warningBox) {

            warningBox.classList.remove(
                "hidden"
            );

            warningBox.style.display =
                "flex";
        }


        const warningTitle =
            $("warningTitle");

        const warningText =
            $("warningText");


        if (warningTitle) {

            warningTitle.textContent =
                "Moderate Risk Estimate";
        }


        if (warningText) {

            warningText.textContent =
                "The model-based estimate is in the MEDIUM band. Consider additional caution when travelling through this area.";
        }


        const warningMessage =
            $("warningMessage");


        if (warningMessage) {

            warningMessage.textContent =
                "The model-based estimate is in the MEDIUM band. Consider additional caution when travelling through this area.";
        }
    }


    if (
        band === "HIGH"
    ) {

        if (warningBox) {

            warningBox.classList.remove(
                "hidden"
            );

            warningBox.style.display =
                "flex";
        }


        const warningTitle =
            $("warningTitle");

        const warningText =
            $("warningText");


        if (warningTitle) {

            warningTitle.textContent =
                "High Risk Estimate";
        }


        if (warningText) {

            warningText.textContent =
                "The model-based estimate is in the HIGH band. Exercise additional caution and consider using the emergency assistance options.";
        }


        const warningMessage =
            $("warningMessage");


        if (warningMessage) {

            warningMessage.textContent =
                "The model-based estimate is in the HIGH band. Exercise additional caution and consider using the emergency assistance options.";
        }


        if (emergencyBox) {

            emergencyBox.classList.remove(
                "hidden"
            );

            emergencyBox.style.display =
                "flex";
        }
    }
}


/* =========================================================
   OUTSIDE COVERAGE
========================================================= */

function displayOutsideCoverage(
    data
) {

    showLoading(false);


    const outsideCard =
        $("outsideCard");


    if (outsideCard) {

        outsideCard.classList.remove(
            "hidden"
        );

        outsideCard.style.display =
            "block";
    }


    if ($("outsideMessage")) {

        $("outsideMessage").textContent =
            data.message ||
            "Your location is outside the currently mapped SafeHer-AI prediction coverage.";
    }


    /*
     * Show the actual GPS coordinates
     * even when risk coverage is unavailable.
     */

    if (
        data.latitude !== undefined &&
        $("latitudeValue")
    ) {

        $("latitudeValue").textContent =
            Number(
                data.latitude
            ).toFixed(6);
    }


    if (
        data.longitude !== undefined &&
        $("longitudeValue")
    ) {

        $("longitudeValue").textContent =
            Number(
                data.longitude
            ).toFixed(6);
    }
}


/* =========================================================
   ERROR
========================================================= */

function showError(
    message
) {

    showLoading(false);

    hideAllResults();


    const errorCard =
        $("errorCard");


    if (errorCard) {

        errorCard.classList.remove(
            "hidden"
        );

        errorCard.style.display =
            "block";
    }


    if ($("errorMessage")) {

        $("errorMessage").textContent =
            message;
    }
}


/* =========================================================
   SOS
========================================================= */

async function triggerSOS() {

    const confirmed =
        confirm(
            "Do you want to record an SOS event for this location?"
        );


    if (!confirmed) {

        return;
    }


    const state =
        $("stateValue")
            .textContent;


    const district =
        $("districtValue")
            .textContent;


    const latitude =
        Number(
            $("latitudeValue")
                .textContent
        );


    const longitude =
        Number(
            $("longitudeValue")
                .textContent
        );


    const riskBand =
        $("riskBand")
            .textContent;


    const riskShare =
        Number(
            $("riskValue")
                .textContent
        );


    try {

        $("sosButton").disabled =
            true;


        $("sosButton").textContent =
            "Recording SOS...";


        const response =
            await fetch(
                `${API_BASE}/sos/trigger`,
                {
                    method: "POST",

                    headers: getAuthHeaders(),

                    body: JSON.stringify({

                        latitude,

                        longitude,

                        risk_band:
                            riskBand,

                        risk_share:
                            riskShare,

                        state,

                        district
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            if (
                response.status === 401
            ) {

                localStorage.removeItem(
                    "safeher_token"
                );

                localStorage.removeItem(
                    "safeher_user"
                );

                currentUser =
                    null;

                stopLocationTracking();

                showAuthentication();

                showLoginForm();

                showAuthMessage(
                    "Your login session has expired. Please login again."
                );

                return;
            }


            throw new Error(
                data.detail ||
                "SOS request failed."
            );
        }


        displaySOSSuccess(
            data
        );


    } catch (error) {

        alert(
            error.message ||
            "Unable to record SOS event."
        );


        $("sosButton").disabled =
            false;


        $("sosButton").innerHTML =
            "<span>🚨</span> SEND SOS";
    }
}


/* =========================================================
   SOS SUCCESS
========================================================= */

function displaySOSSuccess(
    data
) {

    const box =
        $("emergencyBox");


    if (!box) {
        return;
    }


    box.classList.remove(
        "hidden"
    );


    box.style.display =
        "flex";


    box.innerHTML = `

        <div class="emergency-content">

            <div class="emergency-icon">
                ✅
            </div>

            <div>

                <span class="emergency-label">
                    SOS EVENT RECORDED
                </span>

                <h2>
                    Emergency Assistance Ready
                </h2>

                <p>
                    Event ID:
                    <strong>
                        ${
                            data.event_id ||
                            "Recorded"
                        }
                    </strong>
                </p>

            </div>

        </div>


        <div style="
            display:flex;
            gap:10px;
            flex-wrap:wrap;
        ">

            <a
                href="tel:112"
                class="sos-button"
                style="
                    text-decoration:none;
                    display:inline-flex;
                    align-items:center;
                    justify-content:center;
                "
            >
                📞 CALL 112
            </a>


            <button
                type="button"
                class="secondary-btn"
                onclick="shareLocation()"
            >
                📍 SHARE LOCATION
            </button>

        </div>
    `;
}


/* =========================================================
   SHARE LOCATION
========================================================= */

async function shareLocation() {

    const state =
        $("stateValue")
            .textContent;


    const district =
        $("districtValue")
            .textContent;


    const latitude =
        $("latitudeValue")
            .textContent;


    const longitude =
        $("longitudeValue")
            .textContent;


    const text =
        `SafeHer-AI Location\n\n` +
        `State: ${state}\n` +
        `District: ${district}\n` +
        `Latitude: ${latitude}\n` +
        `Longitude: ${longitude}\n\n` +
        `Model-based risk assessment available in SafeHer-AI.`;


    try {

        if (
            navigator.share
        ) {

            await navigator.share({

                title:
                    "SafeHer-AI Location",

                text:
                    text
            });

            return;
        }


        await navigator.clipboard.writeText(
            text
        );


        alert(
            "Location details copied to clipboard."
        );


    } catch (error) {

        console.log(
            "Location sharing cancelled."
        );
    }
}