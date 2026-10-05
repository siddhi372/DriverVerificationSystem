// =====================================================
// DRIVER SECURITY ADMIN PORTAL
// =====================================================

const API = "";

let token = localStorage.getItem("driver_security_token");
let admin = JSON.parse(
    localStorage.getItem("driver_security_admin") || "null"
);

let cameraStream = null;
let capturedImageBlob = null;


// =====================================================
// HELPER
// =====================================================

function $(id) {
    return document.getElementById(id);
}


function showMessage(message, success = false) {

    const element = $("auth-message");

    if (!element) return;

    element.textContent = message;

    element.style.color = success
        ? "#00d4ff"
        : "#ff6b6b";
}


// =====================================================
// LOGIN / REGISTER TABS
// =====================================================

function showLogin() {

    $("login-form").style.display = "block";
    $("register-form").style.display = "none";

    $("login-tab").classList.add("active");
    $("register-tab").classList.remove("active");

    showMessage("");
}


function showRegister() {

    $("login-form").style.display = "none";
    $("register-form").style.display = "block";

    $("login-tab").classList.remove("active");
    $("register-tab").classList.add("active");

    showMessage("");
}


// =====================================================
// LOGIN
// =====================================================

$("login-form").addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        const admin_id =
            $("login-admin-id").value.trim();

        const password =
            $("login-password").value;

        try {

            const response = await fetch(
                "/auth/login",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        admin_id: admin_id,
                        password: password
                    })
                }
            );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Login failed"
                );
            }


            token = data.access_token;
            admin = data.admin;


            localStorage.setItem(
                "driver_security_token",
                token
            );

            localStorage.setItem(
                "driver_security_admin",
                JSON.stringify(admin)
            );


            showDashboard();

        }
        catch (error) {

            showMessage(
                error.message
            );
        }

    }
);


// =====================================================
// OWNER REGISTRATION
// =====================================================

$("register-form").addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        const admin_id =
            $("register-admin-id")
                .value
                .trim();

        const name =
            $("register-name")
                .value
                .trim();

        const password =
            $("register-password")
                .value;


        try {

            const response = await fetch(
                "/auth/register",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        admin_id: admin_id,

                        name: name,

                        password: password
                    })
                }
            );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Registration failed"
                );
            }


            showMessage(
                "Owner account created successfully. Please login.",
                true
            );


            // Switch back to login
            setTimeout(
                function() {

                    showLogin();

                    $("login-admin-id")
                        .value = admin_id;

                },
                1000
            );

        }
        catch (error) {

            showMessage(
                error.message
            );
        }

    }
);


// =====================================================
// SHOW DASHBOARD
// =====================================================

function showDashboard() {

    $("auth-page").style.display =
        "none";

    $("dashboard-page").style.display =
        "block";


    if ($("admin-name")) {

        $("admin-name").textContent =
            admin?.name ||
            admin?.admin_id ||
            "Owner";
    }


    loadDrivers();
    loadVehicles();
}


// =====================================================
// LOGOUT
// =====================================================

function logout() {

    localStorage.removeItem(
        "driver_security_token"
    );

    localStorage.removeItem(
        "driver_security_admin"
    );

    token = null;
    admin = null;

    location.reload();
}


// =====================================================
// DASHBOARD NAVIGATION
// =====================================================

function showSection(section) {

    const sections = [
        "overview",
        "drivers",
        "vehicles",
        "verification",
        "history"
    ];


    sections.forEach(
        function(name) {

            const element =
                $(
                    name + "-section"
                );

            if (element) {

                element.style.display =
                    name === section
                        ? "block"
                        : "none";
            }

        }
    );


    if (section === "drivers") {

        loadDrivers();
    }

    if (section === "vehicles") {

        loadVehicles();
    }
}


// =====================================================
// DRIVER MODAL
// =====================================================

function openDriverModal() {

    $("driver-modal").style.display =
        "flex";

    resetDriverForm();
}


function closeDriverModal() {

    $("driver-modal").style.display =
        "none";

    stopCamera();
}


function resetDriverForm() {

    $("driver-form").reset();

    $("capture-status").textContent =
        "Camera not started.";

    capturedImageBlob = null;

    stopCamera();
}


// =====================================================
// START WEBCAM
// =====================================================

async function startCamera() {

    try {

        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({
                    video: {
                        width: 640,
                        height: 480
                    },
                    audio: false
                });


        $("webcam").srcObject =
            cameraStream;


        $("capture-status").textContent =
            "Camera active. Position one face clearly in the frame.";

    }
    catch (error) {

        $("capture-status").textContent =
            "Unable to access webcam: " +
            error.message;
    }
}


// =====================================================
// STOP CAMERA
// =====================================================

function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;
    }


    if ($("webcam")) {

        $("webcam").srcObject = null;
    }
}


// =====================================================
// CAPTURE FACE
// =====================================================

function captureFace() {

    if (!cameraStream) {

        $("capture-status").textContent =
            "Start the webcam first.";

        return;
    }


    const video =
        $("webcam");

    const canvas =
        $("capture-canvas");


    if (
        video.videoWidth === 0 ||
        video.videoHeight === 0
    ) {

        $("capture-status").textContent =
            "Camera frame is not ready.";

        return;
    }


    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    const context =
        canvas.getContext("2d");


    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );


    canvas.toBlob(
        function(blob) {

            capturedImageBlob =
                blob;


            $("capture-status").textContent =
                "Face image captured successfully. Click REGISTER DRIVER.";

        },
        "image/jpeg",
        0.95
    );
}


// =====================================================
// DRIVER REGISTRATION
// =====================================================

$("driver-form").addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        if (!capturedImageBlob) {

            $("capture-status").textContent =
                "Please capture the driver's face first.";

            return;
        }


        const driverId =
            $("driver-id")
                .value
                .trim();

        const name =
            $("driver-name")
                .value
                .trim();

        const age =
            $("driver-age")
                .value;

        const address =
            $("driver-address")
                .value
                .trim();


        if (!driverId || !name || !age) {

            $("capture-status").textContent =
                "Please fill all required driver details.";

            return;
        }


        const formData =
            new FormData();


        formData.append(
            "driver_id",
            driverId
        );

        formData.append(
            "name",
            name
        );

        formData.append(
            "age",
            age
        );

        formData.append(
            "address",
            address
        );


        formData.append(
            "enrollment_image",
            capturedImageBlob,
            driverId + ".jpg"
        );


        $("capture-status").textContent =
            "AI processing face...";


        try {

            const response =
                await fetch(
                    "/drivers/enroll",
                    {
                        method: "POST",

                        headers: {
                            "Authorization":
                                "Bearer " + token
                        },

                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Driver registration failed"
                );
            }


            $("capture-status").textContent =
                "Driver registered successfully.";


            alert(
                "Driver registered successfully!\n\n" +
                "Driver: " +
                data.driver_name +
                "\nDriver ID: " +
                data.driver_id +
                "\nEmbedding: " +
                data.embedding_dimension +
                " dimensions"
            );


            closeDriverModal();

            loadDrivers();

        }
        catch (error) {

            $("capture-status").textContent =
                "Registration failed: " +
                error.message;
        }

    }
);


// =====================================================
// LOAD DRIVERS
// =====================================================

async function loadDrivers() {

    const container =
        $("drivers-list");

    if (!container) return;


    try {

        const response =
            await fetch(
                "/drivers",
                {
                    headers: {
                        "Authorization":
                            "Bearer " + token
                    }
                }
            );


        const drivers =
            await response.json();


        if (!response.ok) {

            throw new Error(
                drivers.detail ||
                "Unable to load drivers"
            );
        }


        if (!Array.isArray(drivers) ||
            drivers.length === 0) {

            container.innerHTML =
                "<p>No drivers registered yet.</p>";

            return;
        }


        container.innerHTML =
            drivers.map(
                driver => `

                <div class="driver-card">

                    <h3>
                        ${escapeHtml(
                            driver.name
                        )}
                    </h3>

                    <p>
                        Driver ID:
                        ${escapeHtml(
                            driver.driver_id
                        )}
                    </p>

                    <p>
                        Age:
                        ${driver.age ?? "-"}
                    </p>

                    <p>
                        Status:
                        ${escapeHtml(
                            driver.status || "ACTIVE"
                        )}
                    </p>

                </div>

            `
            ).join("");

    }
    catch (error) {

        container.innerHTML =
            "<p>Unable to load drivers.</p>";

        console.error(error);
    }
}


// =====================================================
// LOAD VEHICLES
// =====================================================

async function loadVehicles() {

    const container =
        $("vehicles-list");

    if (!container) return;


    try {

        const response =
            await fetch(
                "/vehicles",
                {
                    headers: {
                        "Authorization":
                            "Bearer " + token
                    }
                }
            );


        if (!response.ok) {

            throw new Error(
                "Unable to load vehicles"
            );
        }


        const vehicles =
            await response.json();


        if (!Array.isArray(vehicles) ||
            vehicles.length === 0) {

            container.innerHTML =
                "<p>No vehicles registered yet.</p>";

            return;
        }


        container.innerHTML =
            vehicles.map(
                vehicle => `

                <div class="driver-card">

                    <h3>
                        Vehicle
                    </h3>

                    <p>
                        Vehicle ID:
                        ${vehicle.vehicle_id}
                    </p>

                    <p>
                        ESP:
                        ${escapeHtml(
                            vehicle.esp_id || "-"
                        )}
                    </p>

                    <p>
                        Number Plate:
                        ${escapeHtml(
                            vehicle.number_plate || "-"
                        )}
                    </p>

                </div>

            `
            ).join("");

    }
    catch (error) {

        container.innerHTML =
            "<p>Unable to load vehicles.</p>";

        console.error(error);
    }
}


// =====================================================
// ESCAPE HTML
// =====================================================

function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


// =====================================================
// INITIALIZATION
// =====================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        if (token && admin) {

            showDashboard();

        } else {

            $("auth-page").style.display =
                "block";

            $("dashboard-page").style.display =
                "none";

            showLogin();
        }

    }
);