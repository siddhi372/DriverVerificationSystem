import tkinter as tk
from tkinter import ttk, messagebox
import requests
import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis


# =====================================================
# CONFIGURATION
# =====================================================

API_URL = "http://127.0.0.1:8000"

access_token = None
admin_info = None

# =====================================================
# FACE ENROLLMENT MODEL
# =====================================================

face_app = None


def get_face_app():
    global face_app

    if face_app is None:
        print("Loading ArcFace enrollment model...")

        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        face_app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("ArcFace enrollment model ready.")

    return face_app


# =====================================================
# LOGIN WINDOW
# =====================================================

def show_login():

    login = tk.Tk()

    login.title("Driver Security - Admin Login")
    login.geometry("500x400")
    login.configure(bg="#111111")
    login.resizable(False, False)


    # -------------------------------------------------
    # TITLE
    # -------------------------------------------------

    tk.Label(
        login,
        text="DRIVER SECURITY",
        font=("Arial", 24, "bold"),
        fg="white",
        bg="#111111"
    ).pack(pady=(40, 5))


    tk.Label(
        login,
        text="ADMIN PORTAL",
        font=("Arial", 16),
        fg="#00ff88",
        bg="#111111"
    ).pack(pady=(0, 30))


    # -------------------------------------------------
    # ADMIN ID
    # -------------------------------------------------

    tk.Label(
        login,
        text="Admin ID",
        font=("Arial", 12),
        fg="white",
        bg="#111111"
    ).pack()

    admin_entry = tk.Entry(
        login,
        font=("Arial", 14),
        width=30,
        bg="#1e1e1e",
        fg="white",
        insertbackground="white"
    )

    admin_entry.pack(pady=8)


    # -------------------------------------------------
    # PASSWORD
    # -------------------------------------------------

    tk.Label(
        login,
        text="Password",
        font=("Arial", 12),
        fg="white",
        bg="#111111"
    ).pack()

    password_entry = tk.Entry(
        login,
        font=("Arial", 14),
        width=30,
        show="*",
        bg="#1e1e1e",
        fg="white",
        insertbackground="white"
    )

    password_entry.pack(pady=8)


    # -------------------------------------------------
    # LOGIN FUNCTION
    # -------------------------------------------------

    def login_admin():

        global access_token
        global admin_info

        admin_id = admin_entry.get().strip()
        password = password_entry.get()


        if not admin_id or not password:

            messagebox.showwarning(
                "Login",
                "Please enter Admin ID and Password."
            )

            return


        try:

            response = requests.post(

                API_URL + "/auth/login",

                json={
                    "admin_id": admin_id,
                    "password": password
                },

                timeout=5
            )


            if response.status_code == 200:

                data = response.json()

                access_token = data.get(
                    "access_token"
                )

                admin_info = data.get(
                    "admin",
                    {}
                )


                messagebox.showinfo(
                    "Login Successful",
                    "Welcome, "
                    + admin_info.get(
                        "name",
                        "Admin"
                    )
                )


                login.destroy()

                open_dashboard()


            elif response.status_code == 401:

                messagebox.showerror(
                    "Login Failed",
                    "Invalid Admin ID or Password."
                )


            elif response.status_code == 403:

                messagebox.showerror(
                    "Access Denied",
                    "This admin account is disabled."
                )


            else:

                messagebox.showerror(
                    "Server Error",
                    f"Server returned {response.status_code}"
                )


        except requests.exceptions.ConnectionError:

            messagebox.showerror(
                "Connection Error",
                "Cannot connect to backend.\n\n"
                "Start FastAPI using:\n"
                "uvicorn server.main:app --reload"
            )


        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


    # -------------------------------------------------
    # LOGIN BUTTON
    # -------------------------------------------------

    tk.Button(
        login,
        text="LOGIN",
        command=login_admin,
        font=("Arial", 14, "bold"),
        bg="#00aa66",
        fg="white",
        activebackground="#00cc77",
        activeforeground="white",
        width=20,
        height=2
    ).pack(pady=25)


    password_entry.bind(
        "<Return>",
        lambda event: login_admin()
    )


    login.mainloop()


# =====================================================
# ADMIN DASHBOARD
# =====================================================

def open_dashboard():

    root = tk.Tk()

    root.title(
        "Driver Security - Admin Portal"
    )

    root.geometry(
        "1100x700"
    )

    root.configure(
        bg="#111111"
    )


    # =================================================
    # HEADER
    # =================================================

    header = tk.Frame(
        root,
        bg="#111111"
    )

    header.pack(
        fill="x",
        padx=30,
        pady=20
    )


    tk.Label(
        header,
        text="DRIVER SECURITY ADMIN PORTAL",
        font=("Arial", 24, "bold"),
        fg="white",
        bg="#111111"
    ).pack(
        side="left"
    )


    # -------------------------------------------------
    # ADMIN INFORMATION
    # -------------------------------------------------

    admin_name = admin_info.get(
        "name",
        "Admin"
    )

    admin_role = admin_info.get(
        "role",
        "ADMIN"
    )


    tk.Label(
        header,
        text=f"{admin_name}  |  {admin_role}",
        font=("Arial", 11),
        fg="#00ff88",
        bg="#111111"
    ).pack(
        side="right"
    )


    # =================================================
    # SYSTEM STATUS
    # =================================================

    status_frame = tk.Frame(
        root,
        bg="#1e1e1e"
    )

    status_frame.pack(
        fill="x",
        padx=30,
        pady=10
    )


    tk.Label(
        status_frame,
        text="SYSTEM STATUS",
        font=("Arial", 16, "bold"),
        fg="white",
        bg="#1e1e1e"
    ).pack(
        pady=10
    )


    status_label = tk.Label(
        status_frame,
        text="Checking backend...",
        font=("Arial", 14),
        fg="yellow",
        bg="#1e1e1e"
    )

    status_label.pack(
        pady=10
    )


    # =================================================
    # SUMMARY
    # =================================================

    summary_frame = tk.Frame(
        root,
        bg="#111111"
    )

    summary_frame.pack(
        fill="x",
        padx=30,
        pady=20
    )


    def create_card(
        parent,
        title,
        value
    ):

        frame = tk.Frame(
            parent,
            bg="#1e1e1e",
            width=200,
            height=120
        )

        frame.pack(
            side="left",
            padx=10,
            expand=True,
            fill="both"
        )

        frame.pack_propagate(False)


        tk.Label(
            frame,
            text=title,
            font=("Arial", 12),
            fg="white",
            bg="#1e1e1e"
        ).pack(
            pady=(20, 5)
        )


        value_label = tk.Label(
            frame,
            text=value,
            font=("Arial", 26, "bold"),
            fg="#00ff88",
            bg="#1e1e1e"
        )

        value_label.pack()

        return value_label


    total_label = create_card(
        summary_frame,
        "TOTAL",
        "0"
    )

    allowed_label = create_card(
        summary_frame,
        "ALLOWED",
        "0"
    )

    denied_label = create_card(
        summary_frame,
        "DENIED",
        "0"
    )

    live_label = create_card(
        summary_frame,
        "LIVE",
        "0"
    )


    # =================================================
    # UPDATE DASHBOARD
    # =================================================

    def update_dashboard():

        try:

            response = requests.get(
                API_URL +
                "/verification/summary",
                timeout=3
            )


            if response.status_code == 200:

                data = response.json()

                summary = data.get(
                    "summary",
                    {}
                )


                total_label.config(
                    text=summary.get(
                        "total_verifications",
                        0
                    )
                )


                allowed_label.config(
                    text=summary.get(
                        "allowed",
                        0
                    )
                )


                denied_label.config(
                    text=summary.get(
                        "denied",
                        0
                    )
                )


                live_label.config(
                    text=summary.get(
                        "live_attempts",
                        0
                    )
                )


                status_label.config(
                    text="● BACKEND ONLINE",
                    fg="#00ff88"
                )


            else:

                status_label.config(
                    text="● BACKEND ERROR",
                    fg="red"
                )


        except Exception:

            status_label.config(
                text="● BACKEND OFFLINE",
                fg="red"
            )


        root.after(
            3000,
            update_dashboard
        )


    # =================================================
    # VERIFICATION HISTORY
    # =================================================

    tk.Label(
        root,
        text="VERIFICATION HISTORY",
        font=("Arial", 18, "bold"),
        fg="white",
        bg="#111111"
    ).pack(
        pady=(10, 5)
    )


    history_frame = tk.Frame(
        root,
        bg="#1e1e1e"
    )

    history_frame.pack(
        fill="both",
        expand=True,
        padx=30,
        pady=10
    )


    columns = (
        "ID",
        "Time",
        "Driver",
        "Quality",
        "Liveness",
        "Confidence",
        "Decision",
        "Vehicle"
    )


    history_table = ttk.Treeview(
        history_frame,
        columns=columns,
        show="headings",
        height=10
    )


    for column in columns:

        history_table.heading(
            column,
            text=column
        )

        history_table.column(
            column,
            width=120,
            anchor="center"
        )


    history_table.pack(
        fill="both",
        expand=True
    )


    # =================================================
    # UPDATE HISTORY
    # =================================================

    def update_history():

        try:

            response = requests.get(
                API_URL +
                "/verification/history",
                timeout=3
            )


            if response.status_code == 200:

                data = response.json()

                history = data.get(
                    "history",
                    []
                )


                for item in history_table.get_children():

                    history_table.delete(
                        item
                    )


                for record in history:

                    confidence = record.get(
                        "driver_confidence",
                        0
                    )


                    try:

                        confidence = (
                            f"{float(confidence):.2f}"
                        )

                    except:

                        confidence = "0.00"


                    history_table.insert(

                        "",

                        "end",

                        values=(

                            record.get(
                                "id",
                                ""
                            ),

                            record.get(
                                "timestamp",
                                ""
                            ),

                            record.get(
                                "driver_name",
                                "UNKNOWN"
                            ),

                            record.get(
                                "image_quality",
                                ""
                            ),

                            record.get(
                                "liveness",
                                ""
                            ),

                            confidence,

                            record.get(
                                "decision",
                                ""
                            ),

                            record.get(
                                "vehicle_status",
                                ""
                            )

                        )
                    )


        except Exception as e:

            print(
                "History error:",
                e
            )


        root.after(
            5000,
            update_history
        )



    # =================================================
    # DRIVER MANAGEMENT
    # =================================================

    def open_drivers_window():

        drivers_window = tk.Toplevel(root)
        drivers_window.title("Driver Management")
        drivers_window.geometry("900x600")
        drivers_window.configure(bg="#111111")

        tk.Label(
            drivers_window,
            text="DRIVER MANAGEMENT",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#111111"
        ).pack(pady=20)

        table_frame = tk.Frame(
            drivers_window,
            bg="#1e1e1e"
        )
        table_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=10
        )

        columns = (
            "ID",
            "Driver ID",
            "Name",
            "Status",
            "Vehicle",
            "Created"
        )

        table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings"
        )

        for column in columns:
            table.heading(column, text=column)
            table.column(
                column,
                width=135,
                anchor="center"
            )

        table.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        def load_drivers():

            try:
                headers = {}

                if access_token:
                    headers["Authorization"] = (
                        "Bearer " + access_token
                    )

                response = requests.get(
                    API_URL + "/drivers",
                    headers=headers,
                    timeout=5
                )

                if response.status_code != 200:
                    messagebox.showerror(
                        "Drivers",
                        f"Server returned {response.status_code}",
                        parent=drivers_window
                    )
                    return

                data = response.json()
                drivers = data.get("drivers", [])

                for item in table.get_children():
                    table.delete(item)

                for driver in drivers:
                    table.insert(
                        "",
                        "end",
                        values=(
                            driver.get("id", ""),
                            driver.get("driver_id", ""),
                            driver.get("name", ""),
                            driver.get("status", ""),
                            driver.get("assigned_vehicle") or "-",
                            driver.get("created_at", "")
                        )
                    )

            except Exception as e:
                messagebox.showerror(
                    "Error",
                    str(e),
                    parent=drivers_window
                )

        def add_driver():
            dialog = tk.Toplevel(drivers_window)
            dialog.title("Register Driver")
            dialog.geometry("520x650")
            dialog.configure(bg="#111111")
            dialog.resizable(False, False)

            tk.Label(
                dialog,
                text="REGISTER DRIVER",
                font=("Arial", 22, "bold"),
                fg="white",
                bg="#111111"
            ).pack(pady=20)

            # -------------------------------------------------
            # DRIVER DETAILS
            # -------------------------------------------------

            form = tk.Frame(
                dialog,
                bg="#111111"
            )
            form.pack(
                fill="x",
                padx=40
            )

            entries = {}

            for label_text, key in (
                ("Driver ID", "driver_id"),
                ("Driver Name", "name"),
                ("Address", "address"),
                ("Age", "age")
            ):

                tk.Label(
                    form,
                    text=label_text,
                    font=("Arial", 11),
                    fg="white",
                    bg="#111111"
                ).pack(
                    anchor="w",
                    pady=(5, 2)
                )

                entry = tk.Entry(
                    form,
                    width=35,
                    font=("Arial", 13),
                    bg="#1e1e1e",
                    fg="white",
                    insertbackground="white"
                )

                entry.pack(
                    fill="x",
                    pady=(0, 8)
                )

                entries[key] = entry

            # -------------------------------------------------
            # FACE STATUS
            # -------------------------------------------------

            face_status = tk.Label(
                dialog,
                text="Face: Not Captured",
                font=("Arial", 12, "bold"),
                fg="yellow",
                bg="#111111"
            )

            face_status.pack(
                pady=(15, 5)
            )

            captured_embedding = None
            captured_image = None

            # -------------------------------------------------
            # CAPTURE FACE
            # -------------------------------------------------

            def capture_face():

                nonlocal captured_embedding
                nonlocal captured_image

                camera = cv2.VideoCapture(
                    0,
                    cv2.CAP_DSHOW
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    640
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    480
                )

                if not camera.isOpened():

                    messagebox.showerror(
                        "Camera Error",
                        "Could not access the webcam.",
                        parent=dialog
                    )

                    return

                try:

                    app = get_face_app()

                    window_name = "Driver Enrollment - Press SPACE"

                    cv2.namedWindow(
                        window_name,
                        cv2.WINDOW_NORMAL
                    )

                    while True:

                        success, frame = camera.read()

                        if not success:
                            break

                        faces = app.get(frame)

                        display = frame.copy()

                        # -------------------------------------
                        # NO FACE
                        # -------------------------------------

                        if len(faces) == 0:

                            cv2.putText(
                                display,
                                "NO FACE DETECTED",
                                (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                            instruction = (
                                "Show your face clearly"
                            )

                        # -------------------------------------
                        # MULTIPLE FACES
                        # -------------------------------------

                        elif len(faces) > 1:

                            cv2.putText(
                                display,
                                "ONLY ONE FACE ALLOWED",
                                (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                            instruction = (
                                "Only one person should be visible"
                            )

                        # -------------------------------------
                        # ONE FACE
                        # -------------------------------------

                        else:

                            face = faces[0]

                            x1, y1, x2, y2 = (
                                face.bbox.astype(int)
                            )

                            cv2.rectangle(
                                display,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                display,
                                "FACE DETECTED",
                                (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2
                            )

                            instruction = (
                                "Press SPACE to capture"
                            )

                        cv2.putText(
                            display,
                            instruction,
                            (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            display,
                            "Q = Cancel",
                            (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (200, 200, 200),
                            2
                        )

                        cv2.imshow(
                            window_name,
                            display
                        )

                        key = cv2.waitKey(1) & 0xFF

                        # -------------------------------------
                        # CANCEL
                        # -------------------------------------

                        if key == ord("q"):

                            break

                        # -------------------------------------
                        # CAPTURE
                        # -------------------------------------

                        if key == 32:

                            if len(faces) != 1:

                                continue

                            face = faces[0]

                            embedding = face.embedding

                            if embedding is None:
                                continue

                            embedding = np.asarray(
                                embedding,
                                dtype=np.float32
                            )

                            norm = np.linalg.norm(
                                embedding
                            )

                            if norm == 0:
                                continue

                            embedding = (
                                embedding / norm
                            )

                            captured_embedding = (
                                embedding.tolist()
                            )

                            captured_image = frame.copy()

                            break

                finally:

                    camera.release()

                    cv2.destroyAllWindows()

                if captured_embedding is not None:

                    face_status.config(
                        text="Face: CAPTURED ✓",
                        fg="#00ff88"
                    )

                else:

                    face_status.config(
                        text="Face: Not Captured",
                        fg="yellow"
                    )

            # -------------------------------------------------
            # CAPTURE BUTTON
            # -------------------------------------------------

            tk.Button(
                dialog,
                text="CAPTURE FACE",
                command=capture_face,
                font=("Arial", 12, "bold"),
                bg="#0066aa",
                fg="white",
                width=20,
                height=2
            ).pack(
                pady=15
            )

            # -------------------------------------------------
            # REGISTER DRIVER
            # -------------------------------------------------

            def save():

                nonlocal captured_embedding
                nonlocal captured_image

                driver_id = (
                    entries["driver_id"]
                    .get()
                    .strip()
                )

                name = (
                    entries["name"]
                    .get()
                    .strip()
                )

                address = (
                    entries["address"]
                    .get()
                    .strip()
                )

                age_text = (
                    entries["age"]
                    .get()
                    .strip()
                )

                # -----------------------------------------
                # VALIDATION
                # -----------------------------------------

                if not driver_id or not name:

                    messagebox.showwarning(
                        "Register Driver",
                        "Driver ID and Name are required.",
                        parent=dialog
                    )

                    return

                if not age_text:

                    messagebox.showwarning(
                        "Register Driver",
                        "Age is required.",
                        parent=dialog
                    )

                    return

                try:

                    age = int(age_text)

                except ValueError:

                    messagebox.showwarning(
                        "Register Driver",
                        "Age must be a number.",
                        parent=dialog
                    )

                    return

                if captured_embedding is None:

                    messagebox.showwarning(
                        "Register Driver",
                        "Please capture the driver's face first.",
                        parent=dialog
                    )

                    return

                # -----------------------------------------
                # CONVERT IMAGE TO JPEG
                # -----------------------------------------

                success, encoded_image = cv2.imencode(
                    ".jpg",
                    captured_image
                )

                if not success:

                    messagebox.showerror(
                        "Registration",
                        "Could not prepare captured image.",
                        parent=dialog
                    )

                    return

                image_bytes = encoded_image.tobytes()

                # -----------------------------------------
                # SEND TO BACKEND
                # -----------------------------------------

                try:

                    headers = {}

                    if access_token:

                        headers["Authorization"] = (
                            "Bearer " + access_token
                        )

                    data = {
                        "driver_id": driver_id,
                        "name": name,
                        "address": address,
                        "age": str(age),
                        "face_embedding": json.dumps(
                            captured_embedding
                        )
                    }

                    files = {
                        "enrollment_image": (
                            f"{driver_id}.jpg",
                            image_bytes,
                            "image/jpeg"
                        )
                    }

                    response = requests.post(
                        API_URL + "/drivers/enroll",
                        headers=headers,
                        data=data,
                        files=files,
                        timeout=30
                    )

                    if response.status_code == 200:

                        messagebox.showinfo(
                            "Success",
                            "Driver registered successfully!\n\n"
                            "Face embedding and enrollment image "
                            "saved to the database.",
                            parent=dialog
                        )

                        dialog.destroy()

                        load_drivers()

                    else:

                        try:

                            detail = response.json().get(
                                "detail",
                                "Registration failed."
                            )

                        except Exception:

                            detail = response.text

                        messagebox.showerror(
                            "Registration Failed",
                            str(detail),
                            parent=dialog
                        )

                except requests.exceptions.ConnectionError:

                    messagebox.showerror(
                        "Connection Error",
                        "Cannot connect to backend.",
                        parent=dialog
                    )

                except Exception as e:

                    messagebox.showerror(
                        "Registration Error",
                        str(e),
                        parent=dialog
                    )

            tk.Button(
                dialog,
                text="REGISTER DRIVER",
                command=save,
                font=("Arial", 12, "bold"),
                bg="#00aa66",
                fg="white",
                width=20,
                height=2
            ).pack(
                pady=15
            )

        def change_status(status):

            selected = table.selection()

            if not selected:
                messagebox.showwarning(
                    "Driver",
                    "Select a driver first.",
                    parent=drivers_window
                )
                return

            values = table.item(
                selected[0],
                "values"
            )

            driver_id = values[1]

            try:
                headers = {
                    "Content-Type": "application/json"
                }

                if access_token:
                    headers["Authorization"] = (
                        "Bearer " + access_token
                    )

                response = requests.put(
                    API_URL +
                    f"/drivers/{driver_id}/status",
                    headers=headers,
                    json={"status": status},
                    timeout=5
                )

                if response.status_code == 200:
                    load_drivers()
                else:
                    try:
                        detail = response.json().get(
                            "detail",
                            "Unable to update status."
                        )
                    except Exception:
                        detail = response.text

                    messagebox.showerror(
                        "Update Failed",
                        str(detail),
                        parent=drivers_window
                    )

            except Exception as e:
                messagebox.showerror(
                    "Error",
                    str(e),
                    parent=drivers_window
                )

        def assign_vehicle():

            selected = table.selection()

            if not selected:
                messagebox.showwarning(
                    "Driver",
                    "Select a driver first.",
                    parent=drivers_window
                )
                return

            values = table.item(
                selected[0],
                "values"
            )

            driver_id = values[1]

            dialog = tk.Toplevel(drivers_window)
            dialog.title("Assign Vehicle")
            dialog.geometry("400x220")
            dialog.configure(bg="#111111")
            dialog.resizable(False, False)

            tk.Label(
                dialog,
                text=f"Vehicle for {driver_id}",
                font=("Arial", 16, "bold"),
                fg="white",
                bg="#111111"
            ).pack(pady=20)

            vehicle_entry = tk.Entry(
                dialog,
                width=28,
                font=("Arial", 13),
                bg="#1e1e1e",
                fg="white",
                insertbackground="white"
            )
            vehicle_entry.pack(pady=10)

            def save_vehicle():

                vehicle = vehicle_entry.get().strip()

                if not vehicle:
                    messagebox.showwarning(
                        "Vehicle",
                        "Enter a vehicle ID.",
                        parent=dialog
                    )
                    return

                try:
                    headers = {
                        "Content-Type": "application/json"
                    }

                    if access_token:
                        headers["Authorization"] = (
                            "Bearer " + access_token
                        )

                    response = requests.put(
                        API_URL +
                        f"/drivers/{driver_id}/vehicle",
                        headers=headers,
                        json={
                            "assigned_vehicle": vehicle
                        },
                        timeout=5
                    )

                    if response.status_code == 200:
                        dialog.destroy()
                        load_drivers()
                    else:
                        try:
                            detail = response.json().get(
                                "detail",
                                "Unable to assign vehicle."
                            )
                        except Exception:
                            detail = response.text

                        messagebox.showerror(
                            "Vehicle Update Failed",
                            str(detail),
                            parent=dialog
                        )

                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        str(e),
                        parent=dialog
                    )

            tk.Button(
                dialog,
                text="SAVE",
                command=save_vehicle,
                font=("Arial", 12, "bold"),
                bg="#0066aa",
                fg="white",
                width=15
            ).pack(pady=15)

        buttons = tk.Frame(
            drivers_window,
            bg="#111111"
        )
        buttons.pack(
            fill="x",
            padx=25,
            pady=15
        )

        tk.Button(
            buttons,
            text="ADD DRIVER",
            command=add_driver,
            font=("Arial", 11, "bold"),
            bg="#00aa66",
            fg="white",
            width=15
        ).pack(side="left", padx=5)

        tk.Button(
            buttons,
            text="ACTIVATE",
            command=lambda: change_status("ACTIVE"),
            font=("Arial", 11, "bold"),
            bg="#006b45",
            fg="white",
            width=13
        ).pack(side="left", padx=5)

        tk.Button(
            buttons,
            text="DEACTIVATE",
            command=lambda: change_status("INACTIVE"),
            font=("Arial", 11, "bold"),
            bg="#aa3333",
            fg="white",
            width=13
        ).pack(side="left", padx=5)

        tk.Button(
            buttons,
            text="ASSIGN VEHICLE",
            command=assign_vehicle,
            font=("Arial", 11, "bold"),
            bg="#0066aa",
            fg="white",
            width=16
        ).pack(side="left", padx=5)

        tk.Button(
            buttons,
            text="REFRESH",
            command=load_drivers,
            font=("Arial", 11, "bold"),
            bg="#444444",
            fg="white",
            width=12
        ).pack(side="right", padx=5)

        load_drivers()


    # -------------------------------------------------
    # DRIVER MANAGEMENT BUTTON
    # -------------------------------------------------

    tk.Button(
        header,
        text="DRIVERS",
        command=open_drivers_window,
        font=("Arial", 10, "bold"),
        bg="#0066aa",
        fg="white",
        activebackground="#0088cc",
        activeforeground="white",
        width=12
    ).pack(
        side="right",
        padx=(10, 0)
    )


    # =================================================
    # START DASHBOARD
    # =================================================

    update_dashboard()

    update_history()

    root.mainloop()


# =====================================================
# APPLICATION START
# =====================================================

if __name__ == "__main__":

    show_login()