from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from firebase_admin_setup import db
from firebase_admin import db as rtdb  # Realtime Database
from firebase_admin import auth as admin_auth
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from auth_rest import signup as rest_signup, signin as rest_signin, send_password_reset
from datetime import datetime
from google.cloud import storage
from flask import flash
import uuid
import json
import csv
import io
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from llm_service import generate_reply
from flask import abort


# إعداد Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "CHANGE_THIS_SECRET_IN_ENV_OR_CONFIG" 

#------------- 
news_pipeline = joblib.load('news_baseline_pipeline.pkl')
# --- RNN (Baseline 2) ---
RNN_MODEL_PATH = "news_rnn_baseline.keras"
RNN_TOKENIZER_PATH = "news_rnn_tokenizer.pkl"
RNN_MAX_LEN = 600  

rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
rnn_tokenizer = joblib.load(RNN_TOKENIZER_PATH)


def split_into_3_chunks(text):
    words = text.split()
    if len(words) <= 3:
        return [text]
    chunk_size = len(words) // 3
    chunks = [
        " ".join(words[:chunk_size]),
        " ".join(words[chunk_size:2*chunk_size]),
        " ".join(words[2*chunk_size:])
    ]
    return chunks

def rnn_predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]

    seq = rnn_tokenizer.texts_to_sequences(texts)
    x = pad_sequences(seq, maxlen=RNN_MAX_LEN, padding="post", truncating="post")

    p_ai = rnn_model.predict(x, verbose=0).ravel()  # numpy array
    p_h = 1.0 - p_ai

    # ✅ رجّعي Python float (مو numpy.float32)
    out = []
    for h, a in zip(p_h, p_ai):
        out.append([float(h), float(a)])
    return out



#------------- 
def get_current_user_doc():
    """
    ترجع وثيقة المستخدم الحالي من Firestore
    بناءً على uid الموجود في الـ session.
    """
    uid = session.get("uid")
    if not uid:
        return None

    snap = db.collection("users").document(uid).get()
    return snap if snap.exists else None


def get_user_full_name(user_doc):
    """
    ترجع الاسم الكامل: firstName + lastName
    لو ما فيه بيانات يرجع 'User'
    """
    if not user_doc:
        return "User"

    data = user_doc.to_dict()
    prof = data.get("profile", {})
    first = prof.get("firstName", "")
    last = prof.get("lastName", "")

    full = f"{first} {last}".strip()
    return full or "User"


# ------------------ صفحات واجهة (GET) ------------------

# 1) استبدلي دالة index() كاملة بهذا الكود
@app.route("/")
def index():
    return render_template("HomePage.html")
    

    uid      = session["uid"]
    user_doc = db.collection("users").document(uid).get()
    role     = user_doc.to_dict().get("role", "user")

@app.route("/login")
def login_page():
    return render_template("Login.html")

@app.route("/signup")
def signup_page():
    return render_template("signup.html")

@app.route("/verified")
def verified():
    # بعد التحقق، نعيد توجيهه لصفحة نجاح التفعيل
    return render_template("Verified.html")

@app.route("/profile")
def profile_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))
    

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template("Profile.html", user_data=user_data, user_name=full_name)

@app.route("/createproject")
def create_project_page():
    project_id = request.args.get("id")  # في حال تم فتح الصفحة للتعديل
    return render_template("CreateProject.html", edit_project_id=project_id)


@app.route("/myprojectowner")
def my_project_owner_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data  = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template("myprojectowner.html", user_name=full_name)
@app.route("/api/add_examiner_to_project", methods=["POST"])
def api_add_examiner_to_project():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")
    data = request.get_json() or {}

    project_id = data.get("project_id")
    examiner_email = data.get("examiner_email")

    if not project_id or not examiner_email:
        return jsonify({"error": "Missing project_id or examiner_email"}), 400

    # نتحقق أن المشروع للمالك الحالي
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # نجيب بيانات الـ Examiner عن طريق الإيميل
    ex_docs = list(
        db.collection("users")
        .where("email", "==", examiner_email)
        .where("role", "==", "examiner")
        .limit(1)
        .stream()
    )

    if not ex_docs:
        return jsonify({"error": "Examiner not found"}), 404

    examiner_uid = ex_docs[0].id
    ex_data = ex_docs[0].to_dict()

    # استخراج اسم examiner
    prof = ex_data.get("profile", {})
    examiner_name = f"{prof.get('firstName','')} {prof.get('lastName','')}".strip()

    # جلب اسم المالك
    owner_doc = db.collection("users").document(owner_uid).get()
    owner_prof = owner_doc.to_dict().get("profile", {})
    owner_name = f"{owner_prof.get('firstName','')} {owner_prof.get('lastName','')}".strip()

    # أوّل شيء نتأكد أنه مو مضاف مسبقًا
    existing = list(
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .limit(1)
        .stream()
    )
    if existing:
        return jsonify({"error": "Examiner already invited"}), 409

    # إنشاء دعوة جديدة
    inv_ref = db.collection("invitations").document()
    inv_ref.set({
        "project_id": project_id,
        "project_name": proj_doc.to_dict().get("project_name"),
        "owner_id": owner_uid,
        "owner_name": owner_name,
        "examiner_id": examiner_uid,
        "examiner_email": examiner_email,
        "status": "pending",  # مباشرة نضيفه مقبول
        "created_at": SERVER_TIMESTAMP
    })

    return jsonify({
        "message": "Examiner added successfully",
        "examiner_name": examiner_name,
        "examiner_email": examiner_email,
        "examiner_id": examiner_uid
    }), 200
    
@app.route("/api/remove_examiner", methods=["POST"])
def api_remove_examiner():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")
    data = request.get_json() or {}

    project_id = data.get("project_id")
    examiner_id = data.get("examiner_id")

    if not project_id or not examiner_id:
        return jsonify({"error": "Missing fields"}), 400

    # تأكيد أن المشروع ملك للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # البحث عن الدعوة المقبولة
    inv_query = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_id)
        .where("project_id", "==", project_id)
        .limit(1)
        .stream()
    )

    inv_list = list(inv_query)
    if not inv_list:
        return jsonify({"error": "Examiner not assigned"}), 404

    inv_id = inv_list[0].id

    # حذف الدعوة
    db.collection("invitations").document(inv_id).delete()

    # حذف الـ examiner من المهام
    tasks = db.collection("tasks").where("project_ID", "==", project_id).stream()
    batch = db.batch()

    for t in tasks:
        t_data = t.to_dict()
        examiners = t_data.get("examiner_ids", [])
        if examiner_id in examiners:
            new_list = [e for e in examiners if e != examiner_id]
            batch.update(
                db.collection("tasks").document(t.id),
                {"examiner_ids": new_list}
            )

    batch.commit()

    return jsonify({"success": True, "message": "Examiner removed"}), 200


@app.route("/myprojectexaminer")
def myprojectexaminer_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data  = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template("myprojectexaminer.html", user_name=full_name)


@app.route("/ownerdashboard")
def owner_dashboard_page():
    # 1) نتحقق أن المستخدم مسجل دخول
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    # 2) نجيب الـ UID من الـ session
    uid = session.get("uid")

    # 3) نجيب بيانات المستخدم من Firestore
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    # 4) نستخرج الاسم
    user_data  = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    # 5) نرسل الاسم للصفحة
    return render_template("Ownerdashboard.html", user_name=full_name)

@app.route("/examinerdashboard")
def examiner_dashboard_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name = user_data.get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"

    return render_template("Examinerdashboard.html", user_name=full_name)

@app.route("/projectdetailsowner/<project_id>")
def project_details_owner(project_id):
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    owner_uid = session["uid"]

    # نتحقق أن المشروع فعلاً للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        abort(404)

    proj_data = proj_doc.to_dict()
    if proj_data.get("owner_id") != owner_uid:
        abort(403)

    # نجيب بيانات المستخدم لعرض الاسم في الهيدر
    user_doc = db.collection("users").document(owner_uid).get()
    if not user_doc.exists:
        abort(404)

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template(
        "ProjectDetailsOwner.html",
        user_name=full_name,
        project_id=project_id
    )
    
@app.route("/api/project_json_owner/<project_id>")
def api_project_json_owner(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session["uid"]

    # نتأكد أن المشروع للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj = proj_doc.to_dict()
    if proj.get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # 🔥 نجيب معلومات الـ Owner (نفس طريقة examiner)
    owner_doc = db.collection("users").document(owner_uid).get()
    if not owner_doc.exists:
        return jsonify({"error": "Owner not found"}), 404

    owner_data = owner_doc.to_dict()
    prof = owner_data.get("profile", {})

    owner_name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip()
    owner_email = owner_data.get("email", "")

    # 🔥 نجيب عدد المقبولين
    accepted_count = sum(
        1
        for _ in db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    return jsonify({
        "project_name": proj.get("project_name"),
        "description": proj.get("description"),
        "domain": proj.get("domain", []),
        "category": proj.get("category"),
        "dataset_url": proj.get("dataset_url", ""),
        "examiners_accepted": accepted_count,

        # 🔥🔥 أهم شي أضفناهم:
        "owner_name": owner_name,
        "owner_email": owner_email
    })
    
# ------------- قائمة Examiners المقبولين (للـ Owner) -------------
@app.route("/api/project_examiners_owner/<project_id>")
def api_project_examiners_owner(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session["uid"]

    # تأكيد أن المشروع ملك للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()
    if proj_data.get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # نجيب جميع المقبولين
    accepted = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    examiners = []
    for inv in accepted:
        d = inv.to_dict()
        ex_id = d.get("examiner_id")
        user_doc = db.collection("users").document(ex_id).get()
        if not user_doc.exists:
            continue

        u = user_doc.to_dict()
        prof = u.get("profile", {})

        name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip()
        email = u.get("email", "")

        examiners.append({
            "id": ex_id,
            "name": name,
            "email": email
        })

    return jsonify({"examiners": examiners})

# --------------------------------------------------
# صفحة تفاصيل المشروع للمُقيّم (Examiner)
# --------------------------------------------------
@app.route("/projectdetailsexaminer/<project_id>")
def project_details_examiner(project_id):
   
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    examiner_uid = session["uid"]

    # نتحقق أن الم examiner قبل الدعوة
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        abort(404)  # أو redirect 404 page

    # نجيب بياناته لعرض الاسم بالهيدر
    user_doc = db.collection("users").document(examiner_uid).get()
    if not user_doc.exists:
        abort(404)

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name = user_data.get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"

    return render_template("ProjectDetailsExaminer.html",
                         user_name=full_name,
                         project_id=project_id)
# ------------------ صفحة تفاصيل المشروع للمُقيّم (Examiner) ------------------
# --------------------------------------------------

def _get_owner_info(owner_uid):
    owner_doc = db.collection("users").document(owner_uid).get()
    if not owner_doc.exists:
        return {"name": "Unknown", "email": ""}

    data = owner_doc.to_dict()
    prof = data.get("profile", {})
    name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip()
    email = data.get("email", "")

    return {"name": name, "email": email}

    
@app.route("/api/project_json/<project_id>")
def api_project_json(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    examiner_uid = session["uid"]

    # نتحقق أن ال examiner قبل الدعوة
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        return jsonify({"error": "Project not found or you are not a member"}), 404

    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404
    proj = proj_doc.to_dict()

    # نجيب بيانات الأونر
    owner_info = _get_owner_info(proj["owner_id"])

    # نعدّ عدد الـ examiners الذين قبلوا الدعوة
    accepted_count = sum(
        1
        for _ in db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    return jsonify(
        {
            "project_name": proj.get("project_name"),
            "description": proj.get("description"),
            "owner_name": owner_info["name"],
            "owner_email": owner_info["email"],
            "domain": proj.get("domain", []),
            "category": proj.get("category"),
            "examiners_accepted": accepted_count,
            "dataset_url": proj.get("dataset_url", ""),
        }
    )
@app.route("/feedback")
def feedback_page():
    return render_template("feedback.html")
# ------------- قائمة Examiners المقبولين في مشروع معين -------------
@app.route("/api/project_examiners/<project_id>")
def api_project_examiners(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    # نتحقق أن السائل مقبول هو الآخر
    examiner_uid = session["uid"]
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        return jsonify({"error": "Forbidden"}), 403

    # نجيب كل المقبولين
    accepted = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    examiners = []
    for a in accepted:
        ex_id = a.to_dict().get("examiner_id")
        ex_doc = db.collection("users").document(ex_id).get()
        if not ex_doc.exists:
            continue
        prof = ex_doc.to_dict().get("profile", {})
        name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip() or "Unknown"
        email = ex_doc.to_dict().get("email", "")
        examiners.append({
            "id": ex_id,
            "name": name,
            "email": email,
            "is_you": ex_id == examiner_uid
        })

    return jsonify({"examiners": examiners})
# ============= INVITATIONS APIs =============

@app.route("/invitation")
def invitation_page():
    """صفحة Invitations (GET)"""
    if not session.get("idToken"):
        return redirect(url_for("login_page"))
    uid = session["uid"]
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))
    first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
    last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"
    return render_template("invitation.html", user_name=full_name)

@app.route("/api/invitations", methods=["GET"])
def api_invitations():
    """جلب الدعوات (JSON)"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["uid"]
    docs = (
        db.collection("invitations")
        .where("examiner_id", "==", uid)
        .where("status", "==", "pending")
        .stream()
    )
    out = []
    for d in docs:
        data = d.to_dict()
        out.append({
            "id": d.id,
            "project_name": data.get("project_name"),
            "owner_name": data.get("owner_name"),
            "status": data.get("status"),
        })
    return jsonify({"invitations": out})

@app.route("/api/invitations/<invitation_id>", methods=["PATCH"])
def api_update_invitation(invitation_id):
    """قبول أو رفض دعوة"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session["uid"]
    data = request.get_json() or {}
    new_status = data.get("status", "").strip().lower()  # تنظيف الإدخال

    # ✅ توحيد الحالة إلى accepted / declined
    if new_status in ["accept", "accepted"]:
        new_status = "accepted"
    elif new_status in ["decline", "declined"]:
        new_status = "declined"
    else:
        return jsonify({"error": "Invalid status"}), 400

    inv_ref = db.collection("invitations").document(invitation_id)
    inv_doc = inv_ref.get()
    if not inv_doc.exists:
        return jsonify({"error": "Invitation not found"}), 404
    if inv_doc.to_dict().get("examiner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    inv_ref.update({"status": new_status})
    return jsonify({"message": f"Invitation {new_status}ed successfully"}), 200

@app.route("/api/volunteers", methods=["GET"])
def api_volunteers():
    # نجيب المستخدمين اللي دورهم examiner واللي مفعلين volunteer.optIn
    docs = (
        db.collection("users")
        .where("role", "==", "examiner")
        .where("volunteer.optIn", "==", True)
        .stream()
    )

    volunteers = []
    for d in docs:
        data = d.to_dict()
        prof = data.get("profile", {})
        volunteers.append({
            "name": f"{prof.get('firstName','')} {prof.get('lastName','')}".strip(),
            "handle": "@" + prof.get("firstName","").lower(),
            "email": data.get("email", ""),
            "tag": prof.get("specialization", "Volunteer")
        })

    return jsonify({"volunteers": volunteers})

# ------------------ Examiner Accepted Projects ------------------
@app.route("/api/accepted_projects", methods=["GET"])
def api_accepted_projects():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    examiner_id = session["uid"]

    invitations = db.collection("invitations") \
        .where("examiner_id", "==", examiner_id) \
        .where("status", "==", "accepted") \
        .stream()

    projects = []
    for inv in invitations:
        inv_data = inv.to_dict()
        project_id = inv_data.get("project_id")
        project_doc = db.collection("projects").document(project_id).get()
        if project_doc.exists:
            proj = project_doc.to_dict()
            projects.append({
                "project_id": project_id,
                "project_name": proj.get("project_name"),
                "owner_name": inv_data.get("owner_name"),
                "domain": proj.get("domain", []),
                "category": proj.get("category"),
                "status": proj.get("status"),
            })

    return jsonify({"projects": projects})

# ------------------ هنا عشان تطلع المشاريع في صفحة الاونر ماي بروجكت------------------
@app.route("/api/my_projects", methods=["GET"])
def api_my_projects():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    projects_ref = db.collection("projects").where("owner_id", "==", uid).stream()

    projects = []
    for doc in projects_ref:
        data = doc.to_dict()
        project_id = doc.id

        # 🔹 نحسب عدد الـ examiners اللي قبلوا المشروع
        accepted_invitations = db.collection("invitations") \
            .where("project_id", "==", project_id) \
            .where("status", "==", "accepted") \
            .stream()
        accepted_count = sum(1 for _ in accepted_invitations)

        projects.append({
            "project_id": project_id,
            "project_name": data.get("project_name"),
            "domain": data.get("domain", []),
            "category": data.get("category"),
            "examiners": accepted_count,  # ✅ العدد الحقيقي
            "status": data.get("status", "active"),
        })

    return jsonify({"projects": projects})

def ingest_owner_dataset_to_rtdb(category, owner_id, project_id, dataset_id, raw_bytes):
    """
    تخزّن ملف CSV في Realtime Database تحت:
      datasets/uploaded_news أو datasets/uploaded_conversations

    - كل ديتاست لها dataset_id واحد ثابت
    - كل صف داخل الديتاست ينحفظ تحت auto key
    - نستخدم payload للصف كامل زي ما هو من CSV
    """
    if not raw_bytes:
        return 0

    # نحدد الفرع حسب نوع الديتاست
    cat = (category or "").strip().lower()
    if cat in ("news", "article", "articles"):
        branch = "uploaded_news"
    elif cat in ("conversation", "conversations", "chat", "chats"):
        branch = "uploaded_conversations"
    else:
        print(f"[ingest] Unknown category '{category}', skipping RTDB ingest.")
        return 0

    # نقرأ الـ CSV كنص
    text = raw_bytes.decode("utf-8", errors="ignore")
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    base_ref = rtdb.reference("datasets").child(branch).child(dataset_id)
    count = 0

    for row in reader:
        data = {
            "dataset_id": dataset_id,      # 👈 ثابت لكل الصفوف اللي من نفس الديتاست
            "project_id": project_id,
            "owner_id": owner_id,
            "payload": row,                # الصف كامل
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_type": "owner_upload",
        }
        base_ref.push(data)  # auto key من Realtime
        count += 1

    print(f"[ingest] Inserted {count} rows into datasets/{branch} for dataset_id={dataset_id}")
    return count

# ------------------ Create Project (مع إنشاء سجلات invitations منفصلة) ------------------
@app.route("/api/create_project", methods=["POST"])
def api_create_project():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    if not uid:
        return jsonify({"error": "Missing user ID"}), 401

    # نقرأ البيانات سواء من form أو JSON
    data = request.form if request.form else (request.json or {})

    project_name = data.get("project_name")
    description  = data.get("description")
    category     = data.get("category")
    dataset_id = str(uuid.uuid4())
      
 # === منع إنشاء مشروع News بدون Dataset ===
    if category and category.lower() in ["article", "news", "news article"]:
      file_check = request.files.get("dataset")
      if not file_check or not file_check.filename:
        return jsonify({
            "error": "Dataset file is required for News Article projects."
        }), 400


    if hasattr(data, "getlist"):
        domains = data.getlist("domain")
    else:
        domains = data.get("domain", [])

    examiners_raw = data.get("invited_examiners", "[]")
    try:
        examiners = json.loads(examiners_raw) if isinstance(examiners_raw, str) else examiners_raw
    except json.JSONDecodeError:
        examiners = []

    if not project_name or not description or not category:
        return jsonify({"error": "Missing required fields"}), 400

    # نقرأ ملف الديتاست دون تخزينه في Storage
    dataset_url = ""   # ما نستخدم Storage حالياً
    raw_bytes   = None

    file = request.files.get("dataset")
    if file and file.filename:
        raw_bytes = file.read()
        file.seek(0)

    # جلب بيانات الأونر من Firestore
    owner_doc = db.collection("users").document(uid).get()
    if not owner_doc.exists:
        return jsonify({"error": "Owner not found"}), 404

    owner_data = owner_doc.to_dict()
    owner_name = f"{owner_data.get('profile', {}).get('firstName', '')} {owner_data.get('profile', {}).get('lastName', '')}".strip()

    # إنشاء سجل المشروع
    project_id = str(uuid.uuid4())
    project_doc = {
    "project_ID": project_id,
    "project_name": project_name,
    "description": description,
    "domain": domains,
    "category": category,
    "created_at": datetime.utcnow().isoformat() + "Z",
    "owner_id": uid,
    "dataset_id": dataset_id,
    "invited_examiners": [ex.get("email") for ex in examiners],
    "status": "active",
}


    db.collection("projects").document(project_id).set(project_doc)

    # إنشاء الدعوات في Collection منفصل
    batch = db.batch()
    for ex in examiners:
        email = ex.get("email")
        if not email:
            continue

        examiner_docs = list(
            db.collection("users")
              .where("email", "==", email)
              .where("role", "==", "examiner")
              .limit(1)
              .stream()
        )
        if not examiner_docs:
         return jsonify({"error": "Invalid examiner information"}), 400

        examiner_uid = examiner_docs[0].id
        invitation_ref = db.collection("invitations").document()
        invitation_data = {
            "project_id": project_id,
            "project_name": project_name,
            "owner_id": uid,
            "owner_name": owner_name,
            "examiner_id": examiner_uid,
            "status": "pending",
            "created_at": SERVER_TIMESTAMP,
            "examiner_email": email,
        }
        batch.set(invitation_ref, invitation_data)

    if examiners:
        batch.commit()

    # إدخال الديتاست إلى Realtime Database لو فيه ملف
    if raw_bytes:
        try:
            ingest_owner_dataset_to_rtdb(category, uid, project_id, dataset_id, raw_bytes)
        except Exception as e:
            app.logger.exception("Failed to ingest owner dataset into Realtime: %s", e)

    # ✅ في النهاية لازم نرجّع Response واضح دائماً
    return jsonify({
        "message": "Project created successfully",
        "project_ID": project_id,
        "dataset_id": dataset_id,
    }), 201
@app.route("/api/update_project/<project_id>", methods=["POST"])
def api_update_project(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    proj_ref = db.collection("projects").document(project_id)
    proj_doc = proj_ref.get()

    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    data = request.form

    name = data.get("project_name", "").strip()
    desc = data.get("description", "").strip()
    category = data.get("category")
    domains = data.getlist("domain")

    update_data = {
        "project_name": name,
        "description": desc,
        "category": category,
        "domain": domains,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    proj_ref.update(update_data)

    return jsonify({"message": "Project updated successfully"}), 200

# ------------------ حذف مشروع ------------------
@app.route("/api/delete_project/<project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    project_ref = db.collection("projects").document(project_id)
    project_doc = project_ref.get()

    if not project_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if project_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    invitations = db.collection("invitations").where("project_id", "==", project_id).stream()
    batch = db.batch()
    for inv in invitations:
        batch.delete(db.collection("invitations").document(inv.id))
    batch.delete(project_ref)
    batch.commit()

    return jsonify({"message": "Project deleted successfully"}), 200

# ========== إرسال دعوة جديدة للـ Examiner ==========
@app.route("/api/send_invitation", methods=["POST"])
def api_send_invitation():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session["uid"]
    data       = request.get_json() or {}
    examiner_email = data.get("examiner_email")   # أو id حسب تصميمك
    project_id     = data.get("project_id")

    if not examiner_email or not project_id:
        return jsonify({"error": "Missing examiner_email or project_id"}), 400

    # نجيب بيانات الـ Examiner من Firestore بالـ email
    examiner_docs = list(db.collection("users").where("email", "==", examiner_email).limit(1).stream())
    if not examiner_docs:
        return jsonify({"error": "Examiner email not found"}), 404
    examiner_uid = examiner_docs[0].id

    # نجيب بيانات الـ Project للتأكد أنه تابع للـ Owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists or proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Project not found or not yours"}), 403

    # نجيب اسم الـ Owner للعرض
    owner_doc = db.collection("users").document(owner_uid).get()
    owner_name = f"{owner_doc.to_dict()['profile']['firstName']} {owner_doc.to_dict()['profile']['lastName']}".strip()

    # ننشئ الدعوة
    inv_doc = {
        "project_id":   project_id,
        "project_name": proj_doc.to_dict().get("project_name"),
        "owner_id":     owner_uid,
        "owner_name":   owner_name,
        "examiner_id":  examiner_uid,
        "status":       "pending",
        "created_at":   SERVER_TIMESTAMP
    }
    db.collection("invitations").add(inv_doc)

    return jsonify({"message": "Invitation sent successfully"}), 201

# ------------------ مصادقة (POST APIs) ------------------

# إنشاء حساب جديد
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.form if request.form else (request.json or {})

    email    = data.get("email")
    password = data.get("password")

    # 👈 نقرأ اليوزر نيم من الفورم
    username = data.get("username") or data.get("displayName", "")

    role       = data.get("role", "user")
    first_name = data.get("firstName", "")
    last_name  = data.get("lastName", "")
    gender     = data.get("gender", "")
    interests  = data.get("interests", "")
    github     = data.get("github", "")
    linkedin   = data.get("linkedin", "")

    volunteer_opt_in = str(data.get("volunteerOptIn", "false")).lower() == "true"
    specialization   = data.get("specialization", "")
    description      = data.get("description", "")

    # ✅ نتأكد من كل القيم الأساسية
    if not email or not password or not username:
        return jsonify({"error": "email, password and username are required"}), 400

    # ✅ نتأكد أن اليوزر نيم يونيك
    existing = list(
        db.collection("users")
          .where("username", "==", username)
          .limit(1)
          .stream()
    )
    if existing:
        return jsonify({
            "error": "USERNAME_TAKEN",
            "message": "This username is already in use."
        }), 409


    volunteer_opt_in = str(data.get("volunteerOptIn", "false")).lower() == "true"
    specialization   = data.get("specialization", "")
    description      = data.get("description", "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    try:
        res = rest_signup(email, password)  # Firebase
        uid = res["localId"]

        # إرسال رابط التحقق
        send_verification_email(res["idToken"])

        user_doc = {
            "uid": uid,
            "email": email,
             "username": username,  
            "role": role,
            "createdAt": SERVER_TIMESTAMP,
            "updatedAt": SERVER_TIMESTAMP,
            "profile": {
                "firstName": first_name,
                "lastName":  last_name,
                "gender":    gender,
                "interests": interests,
                "github":    github,
                "linkedin":  linkedin,
            }
        }

        if role == "examiner":
            user_doc["profile"]["specialization"] = specialization
            user_doc["profile"]["description"]    = description
            user_doc["volunteer"] = {"optIn": volunteer_opt_in}

        db.collection("users").document(uid).set(user_doc)

        # حفظ بيانات التسجيل مؤقتاً
        session["email"] = email
        session["temp_password"] = password

        # توجيه صفحة التحقق
        return render_template("CheckEmail.html")

    except Exception as e:
        try:
            return jsonify(e.response.json()), e.response.status_code
        except:
            return jsonify({"error": str(e)}), 500

# verification_email هنا كل مايخص
def send_verification_email(id_token):
    url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key=AIzaSyChtQ2FaenDwe7k7bfRB8Cw5G_5C4f_xt4"
    payload = {
        "requestType": "VERIFY_EMAIL",
        "idToken": id_token,
        "continueUrl": "http://127.0.0.1:5000/verified"
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, json=payload, headers=headers)

    print("\n🔥 VERIFY EMAIL RESPONSE 🔥")
    print("Status:", r.status_code)
    print("Body:", r.text)
    print("🔥 ----------------------🔥\n")

    return r

@app.route("/auto-login")
def auto_login():
    email = session.get("email")
    password = session.get("temp_password")

    if not email or not password:
        return redirect(url_for("login_page"))

    try:
        res = rest_signin(email, password)

        session["idToken"] = res["idToken"]
        session["uid"] = res["localId"]

        role = db.collection("users").document(res["localId"]).get().to_dict().get("role")

        if role == "owner":
            return redirect(url_for("owner_dashboard_page"))
        elif role == "examiner":
            return redirect(url_for("examiner_dashboard_page"))
        else:
            return redirect(url_for("profile_page"))

    except:
        return redirect(url_for("login_page"))

# تسجيل الدخول
@app.route("/api/signin", methods=["POST"])
def api_signin():
    data = request.form if request.form else (request.json or {})

    # المستخدم يقدر يكتب email أو username في نفس الحقل
    identifier = (data.get("identifier") or data.get("email") or "").strip()
    password   = data.get("password")

    if not identifier or not password:
        return render_template(
            "Login.html",
            error="Email/username and password are required."
        ), 400

    # 1) نحدّد الإيميل
    email = identifier

    # لو ما فيه @ نفترض أنه Username ونبحث عنه في Firestore
    if "@" not in identifier:
        # تأكدّي أن عندك حقل اسمه "username" داخل وثيقة المستخدم في Firestore
        user_q = (
            db.collection("users")
              .where("username", "==", identifier)
              .limit(1)
              .stream()
        )
        user_docs = list(user_q)
        if not user_docs:
            # Username غير صحيح
            return render_template(
                "Login.html",
                error="Invalid username or password. Please try again."
            ), 401

        user_data = user_docs[0].to_dict()
        email = user_data.get("email")
        if not email:
            return render_template(
                "Login.html",
                error="User record is missing email."
            ), 500

    try:
        # 2) نسجّل الدخول في Firebase Auth باستخدام الإيميل اللي استخرجناه
        res = rest_signin(email, password)

        # 3) نتأكد من تفعيل الإيميل من Firebase (نفس كودك السابق)
        url = "https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=AIzaSyChtQ2FaenDwe7k7bfRB8Cw5G_5C4f_xt4"
        r = requests.post(url, json={"idToken": res["idToken"]})
        user_info = r.json()

        email_verified = user_info["users"][0]["emailVerified"]
        if not email_verified:
            return render_template(
                "Login.html",
                error="Please verify your email before logging in."
            )

        uid = res["localId"]
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            session.clear()
            return render_template(
                "Login.html",
                error="User data not found."
            ), 401

        role = user_doc.to_dict().get("role", "user")

        session["idToken"] = res["idToken"]
        session["uid"] = uid

        if role == "owner":
            return redirect(url_for("owner_dashboard_page"))
        elif role == "examiner":
            return redirect(url_for("examiner_dashboard_page"))
        else:
            return redirect(url_for("profile_page"))

    except Exception:
        app.logger.exception("Signin failed")
        return render_template(
            "Login.html",
            error="Invalid email/username or password. Please try again."
        ), 401


# تسجيل الخروج
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# إرسال رابط إعادة تعيين كلمة المرور
@app.route("/api/reset", methods=["POST"])
def api_reset():
    email = request.form.get("email") or (request.json or {}).get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400
    try:
        send_password_reset(email)
        return jsonify({"message": "Password reset email sent"})
    except Exception as e:
        try:
            return jsonify(e.response.json()), e.response.status_code
        except:
            return jsonify({"error": str(e)}), 500



# صحة الخادم
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/update-profile", methods=["POST"])
def api_update_profile():
    # لازم يكون مسجل دخول
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    user_ref = db.collection("users").document(uid)
    snap = user_ref.get()
    if not snap.exists:
        return jsonify({"error": 'User not found'}), 404

    data = request.get_json() or {}

    first_name     = (data.get("firstName") or "").strip()
    last_name      = (data.get("lastName") or "").strip()
    gender         = (data.get("gender") or "").strip()
    specialization = (data.get("specialization") or "").strip()
    github         = (data.get("github") or "").strip()
    linkedin       = (data.get("linkedin") or "").strip()
    description    = (data.get("description") or "").strip()
    interests      = (data.get("interests") or "").strip()

    updates = {
        "updatedAt": SERVER_TIMESTAMP,
        "profile.firstName": first_name,
        "profile.lastName": last_name,
        "profile.gender": gender,
        "profile.specialization": specialization,
        "profile.github": github,
        "profile.linkedin": linkedin,
        "profile.description": description,
        "profile.interests": interests,
    }

   

    user_ref.update(updates)

    return jsonify({"message": "Profile updated successfully"}), 200



@app.route("/forgot", methods=["GET", "POST"])
def forgot_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        if not email:
            flash("Please enter your email address.", "error")
            return render_template("ForgotPassword.html")

        try:
            # تستخدمين نفس الفنكشن اللي عندك في auth_rest
            send_password_reset(email)
            flash("If this email is registered, we’ve sent a reset link.", "success")
        except Exception as e:
            print("Reset error:", e)
            flash("Something went wrong. Please try again.", "error")

        return render_template("ForgotPassword.html", email=email)

    # GET
    return render_template("ForgotPassword.html")

#CraetaTask
@app.route("/api/create_task", methods=["POST"])
def api_create_task():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    data = request.get_json() or {}
    project_id = data.get("project_id")
    task_name = data.get("task_name")
    examiner_uids = data.get("examiner_ids", [])


    if not project_id or not task_name:
        return jsonify({"error": "Missing required fields"}), 400

    # ---- نجيب بيانات المشروع ----
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()

    # نتأكد أن هذا المشروع ملك للـ Owner الحالي
    if proj_data.get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    category = proj_data.get("category", "").lower()


    if not examiner_uids:
        return jsonify({"error": "No valid examiners selected"}), 400

    # ---- تجهيز بيانات الـ Task ----
    task_id = str(uuid.uuid4())

    task_doc = {
        "task_ID": task_id,
        "project_ID": project_id,
        "task_name": task_name,
        "examiner_ids": examiner_uids,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": owner_uid,
        "status": "pending",
    }

    # ----------- حسب نوع المشروع -----------
    if category == "article":
        # Article Project → نحتاج task_type
        task_type = data.get("task_type")

        if task_type not in ["model_selection", "labeling"]:
            return jsonify({"error": "Invalid task_type. Must be 'model_selection' or 'FeedBack'"}), 400

        task_doc["task_type"] = task_type

        # Model Selection يتطلب examiner واحد فقط
        if task_type == "model_selection" and len(examiner_uids) != 1:
            return jsonify({"error": "Model Selection task requires exactly 1 examiner"}), 400

    else:
        # Conversation Project → نحتاج conversation_type و number_of_turns
        conversation_type = data.get("conversation_type")
        number_of_turns = data.get("number_of_turns")

        if conversation_type not in ["human-ai", "human-human"]:
            return jsonify({"error": "Invalid conversation_type"}), 400

        if not (2 <= number_of_turns <= 7):
            return jsonify({"error": "number_of_turns must be 2–7"}), 400

        task_doc["conversation_type"] = conversation_type
        task_doc["number_of_turns"] = number_of_turns

    # ---- حفظ الـ Task ----
    db.collection("tasks").document(task_id).set(task_doc)

    return jsonify({
        "message": "Task created successfully",
        "task_id": task_id
    }), 201

@app.route("/projects/<project_id>/tasks/create")
def create_task_page(project_id):
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    owner_uid = session.get("uid")
    proj_doc = db.collection("projects").document(project_id).get()

    if not proj_doc.exists:
        abort(404)

    proj_data = proj_doc.to_dict()

    if proj_data.get("owner_id") != owner_uid:
        abort(403)

   
    category = proj_data.get("category", "").lower()


    return render_template("CreateTask.html", project_id=project_id, category=category)

@app.route("/api/project_examiners_for_task/<project_id>")
def get_project_examiners_for_task(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    # تأكيد أن المشروع موجود
    project_doc = db.collection("projects").document(project_id).get()
    if not project_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    # 🟦 نجيب جميع الـ examiners اللي قبلوا الدعوة
    accepted = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    examiners_list = []

    for inv in accepted:
        data = inv.to_dict()
        uid = data.get("examiner_id")

        # جلب بيانات اليوزر
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            info = user_doc.to_dict()
            prof = info.get("profile", {})

            examiners_list.append({
                "uid": uid,
                "email": info.get("email", ""),
                "name": f"{prof.get('firstName','')} {prof.get('lastName','')}".strip()
            })

    return jsonify({"examiners": examiners_list})
@app.route("/api/project_tasks/<project_id>")
def api_project_tasks(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    tasks_ref = (
        db.collection("tasks")
        .where("project_ID", "==", project_id)
        .stream()
    )

    tasks = []
    for t in tasks_ref:
        data = t.to_dict()

        examiner_ids = data.get("examiner_ids", []) or []

        # 🟢 نجمع كل الإيميلات
        examiner_emails = []
        for ex_id in examiner_ids:
            ex_doc = db.collection("users").document(ex_id).get()
            if ex_doc.exists:
                email = ex_doc.to_dict().get("email", "")
                if email:
                    examiner_emails.append(email)

        tasks.append({
            "id": data.get("task_ID"),
            "title": data.get("task_name"),
            "status": data.get("status", "pending"),
            "conversationType": data.get("conversation_type"),
            "turns": data.get("number_of_turns"),
            "examinerCount": len(examiner_emails),
            "primaryExaminerEmail": examiner_emails[0] if examiner_emails else "",
            "examinerEmails": examiner_emails,  
        })

    return jsonify({"tasks": tasks})

# ------------------ Examiner Tasks (Only tasks assigned to this examiner) ------------------
@app.route("/api/examiner_tasks/<project_id>")
def api_examiner_tasks(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    examiner_uid = session.get("uid")

    # تأكيد أن الـ Examiner مقبول في المشروع
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        return jsonify({"error": "Forbidden"}), 403

    tasks_ref = (
        db.collection("tasks")
        .where("project_ID", "==", project_id)
        .stream()
    )

    tasks = []
    for t in tasks_ref:
        data = t.to_dict()

        task_id          = data.get("task_ID")
        conversation_type = data.get("conversation_type", None)
        max_turns         = int(data.get("number_of_turns", 0) or 0)

        examiner_ids = data.get("examiner_ids", []) or []
        assigned = examiner_uid in examiner_ids

        # ✅ نجمع كل إيميلات الممتحنين في الكرت
        examiner_emails = []
        for ex_id in examiner_ids:
            ex_doc = db.collection("users").document(ex_id).get()
            if ex_doc.exists:
                em = ex_doc.to_dict().get("email", "")
                if em:
                    examiner_emails.append(em)

               # -----------------------------
        # 🔵 حساب وضعك أنت على هذا التسك
        # -----------------------------
        personal_status = "pending"
        your_turn = 0

        if assigned and conversation_type in ("human-ai", "human-human") and max_turns > 0:
            try:
                # نجيب كل رسائل هذا التاسك من RTDB
                if conversation_type == "human-ai":
                    conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
                else:
                    conv_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")

                raw = conv_ref.get() or {}

                if isinstance(raw, dict):
                    msgs = list(raw.values())
                elif isinstance(raw, list):
                    msgs = raw
                else:
                    msgs = []

                # -------------------------
                # 👇 حساب عدد التيرنز لك
                # -------------------------
                if conversation_type == "human-ai":
                    # نفس المنطق القديم: كل رسالة من الـ Examiner = 1 turn
                    count_for_me = 0
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue

                        ex_id = m.get("examiner_id") or m.get("sender_id")
                        if ex_id != examiner_uid:
                            continue

                        if m.get("sender_type") != "Ex":
                            continue

                        count_for_me += 1

                    your_turn = count_for_me

                else:
                    # 👈 Human-Human: نستخدم نفس منطق الـ runs
                    # نتأكد من قائمة الـ examiners
                    ex_ids = examiner_ids or list({
                        m.get("examiner_id") or m.get("sender_id")
                        for m in msgs
                        if isinstance(m, dict) and (m.get("examiner_id") or m.get("sender_id"))
                    })

                    # نرتب الرسائل زمنيًا
                    msgs.sort(key=lambda m: m.get("created_at", ""))

                    # نستخدم الفنكشن اللي فوق
                    your_turn = _compute_hh_turns_for_examiner(msgs, examiner_uid, ex_ids)

                # ما نتعدى الحد الأقصى
                if max_turns > 0:
                    your_turn = min(your_turn, max_turns)

            except Exception as e:
                app.logger.exception(
                    "Failed to compute turns for task %s: %s", task_id, e
                )
                your_turn = 0

            if your_turn >= max_turns:
                personal_status = "completed"
            elif your_turn > 0:
                personal_status = "progress"
            else:
                personal_status = "pending"
        else:
            # لو ما هي مهمة محادثة أو مو مسندة لك، نرجع الحالة العامة
            personal_status = data.get("status", "pending")

        tasks.append({
            "task_id": task_id,
            "task_name": data.get("task_name"),
             "task_type": data.get("task_type"),  #CONV OR ART
            # ✅ هذه التي تستخدمها الكروت والفلاتر
            "status": personal_status,

            # الحالة العامة لو احتجتيها
            "global_status": data.get("status", "pending"),

            "conversation_type": conversation_type,
            "number_of_turns": max_turns,
            "current_turn_for_you": your_turn,

            "is_assigned_to_you": assigned,
            "assignment_label": examiner_emails[0] if examiner_emails else "",
            "examiner_emails": examiner_emails,
            "examiner_count": len(examiner_emails),
        })

    return jsonify({"tasks": tasks})


@app.route("/api/tasks/<task_id>/delete", methods=["POST"])
def api_delete_task(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    # نجيب المهمة
    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()

    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict()
    project_id = task_data.get("project_ID")

    # نجيب المشروع للتأكد أن هذا الـ Owner هو صاحب المشروع
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    # 🗑️ حذف المهمة
    task_ref.delete()

    return jsonify({"success": True, "message": "Task deleted successfully"}), 200




@app.route("/api/update_task/<task_id>", methods=["PATCH"])
def api_update_task(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    # نجيب المهمة
    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()

    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict()
    project_id = task_data.get("project_ID")

    # نتأكد أن ال Owner هو صاحب المشروع
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # نقرأ البيانات الجديدة
    data = request.get_json() or {}

    new_name = data.get("task_name", "").strip()
    new_examiners = data.get("examiner_ids", [])

    if not new_name:
        return jsonify({"error": "Task name is required"}), 400

    if not new_examiners:
        return jsonify({"error": "At least one examiner is required"}), 400

    # نحدّث فقط اللي تبينه
    update_data = {
        "task_name": new_name,
        "examiner_ids": new_examiners,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    task_ref.update(update_data)

    return jsonify({"message": "Task updated successfully"}), 200
 
# ===================================================================
# ------------- صفحة Human ↔ AI Conversation (Front) ---------------
# ===================================================================
 
@app.route("/conversation-ai")
def conversation_ai_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))
 
    user_doc = get_current_user_doc()
    user_name = get_user_full_name(user_doc) if user_doc else "User"
 
    # نقرأ taskId من الرابط
    task_id = request.args.get("taskId")

    # 👇 نقرأ project_id من الرابط
    project_id = request.args.get("projectId")
 
    # قيم افتراضية
    max_turns = 6
    task_title = "Conversation task topic"
 
    if task_id:
        try:
            task_snapshot = db.collection("tasks").document(task_id).get()
            if task_snapshot.exists:
                task_data = task_snapshot.to_dict()
                max_turns = int(task_data.get("number_of_turns", 6))
                task_title = task_data.get("task_name", task_title)
        except Exception as e:
            app.logger.exception("Error loading task in conversation_ai_page: %s", e)
 
    return render_template(
        "ConversationH-AI.html",
        user_name=user_name,
        max_turns=max_turns,
        task_title=task_title,
        task_id=task_id,
        project_id=project_id  
)


# ===================================================================
# ------------- صفحة Human ↔ Human Conversation (Front) ------------
# ===================================================================
 
@app.route("/conversation-hh")
def conversation_hh_page():
    # لازم يكون مسجل دخول
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    # نجيب اسم المستخدم
    user_doc = get_current_user_doc()
    user_name = get_user_full_name(user_doc) if user_doc else "User"

    # 🔹 هنا كنا نقرأ بس taskId
    task_id = request.args.get("taskId")
    project_id = request.args.get("projectId")  # <-- ✅ (1) أضفنا قراءة projectId من الكويري

    # قيم افتراضية
    max_turns = 6
    task_title = "Human ↔ Human conversation task"

    if task_id:
        try:
            task_snapshot = db.collection("tasks").document(task_id).get()
            if task_snapshot.exists:
                task_data  = task_snapshot.to_dict()
                max_turns  = int(task_data.get("number_of_turns", 6))
                task_title = task_data.get("task_name", task_title)
                conv_type  = task_data.get("conversation_type")

                # لو طلع النوع مو Human-Human نحوله لصفحة AI زي ما كان
                if conv_type != "human-human":
                    return redirect(
                        url_for("conversation_ai_page", taskId=task_id, projectId=project_id)
                    )
        except Exception as e:
            app.logger.exception("Error loading task in conversation_hh_page: %s", e)

    return render_template(
        "ConversationH-H.html",
        user_name=user_name,
        max_turns=max_turns,
        task_title=task_title,
        task_id=task_id,
        project_id=project_id,  # <-- ✅ (2) نمرر project_id للتمبليت
    )
# ==========================
#  AI Conversation Reply API
# ==========================
@app.route("/api/ai_reply", methods=["POST"])
def api_ai_reply():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    task_id = data.get("taskId")

    if not user_message or not task_id:
        return jsonify({"error": "Missing message or taskId"}), 400

    sender_id = session.get("uid")

    # اسم المستخدم
    sender_doc = db.collection("users").document(sender_id).get()
    sender_name = "User"
    if sender_doc.exists:
        prof = sender_doc.to_dict().get("profile", {})
        sender_name = f"{prof.get('firstName','')} {prof.get('lastName','')}".strip() or "User"

    ref = rtdb.reference(f"llm_conversations/{task_id}/messages")

    existing = ref.get() or {}
    count_user = sum(1 for x in existing.values() if x.get("sender_type") == "Ex") + 1

    turn_id = str(uuid.uuid4())
    now_iso = datetime.utcnow().isoformat() + "Z"

    # 🧍‍♀️ 1) نحفظ رسالة المستخدم
    ref.push({
        "turn_id": turn_id,
        "task_id": task_id,
        "turn_number": count_user,
        "sender_type": "Ex",
        "examiner_id": sender_id,
        "sender_name": sender_name,
        "message": user_message,
        "created_at": now_iso,
    })

    # 🤖 2) نجيب رد AI
    try:
        ai_response = generate_reply(user_message)
    except Exception:
        ai_response = "Sorry, I couldn’t generate a reply."

    # 🧠 3) نحفظ رسالة الـ AI بنفس turn_id
    ref.push({
        "turn_id": turn_id,
        "task_id": task_id,
        "turn_number": count_user,
        "sender_type": "LLM",
        "sender_name": "AI",
        "message": ai_response,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })

    # ✅ 4) نحدّث حالة التاسك لو اكتملت
    _update_ai_task_status_if_completed(task_id)

    return jsonify({"reply": ai_response}), 200

# ==========================
#  AI Conversation message API
# ==========================
@app.route("/api/ai/messages", methods=["GET"])
def api_ai_get_messages():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "Missing taskId"}), 400

    uid = session.get("uid")

    try:
        ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
        raw = ref.get() or {}

        if isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []

        # نرتب بالوقت
        rows.sort(key=lambda m: m.get("created_at", ""))

               # نخلي كل Examiner يشوف محادثته هو فقط
        my_turn_ids = {
            m.get("turn_id")
            for m in rows
            if isinstance(m, dict)
            and (m.get("examiner_id") == uid or m.get("sender_id") == uid)
        }

        messages = []
        your_turn = 0

        for m in rows:
            if not isinstance(m, dict):
                continue
            if m.get("turn_id") not in my_turn_ids:
                continue

            sender_type = m.get("sender_type")
            text = m.get("message", "")

            if sender_type == "LLM":
                side = "ai"
            else:
                side = "you"
                your_turn = max(your_turn, int(m.get("turn_number", 0) or 0))

            messages.append({
                "text": text,
                "side": side,
            })

             # حالة التاسك من Firestore + عدد التيرنز
        task_status = "pending"
        max_turns = 0
        try:
            task_doc = db.collection("tasks").document(task_id).get()
            if task_doc.exists:
                tdata = task_doc.to_dict()
                task_status = tdata.get("status", "pending")
                max_turns = int(tdata.get("number_of_turns", 0) or 0)

                # ✅ لو مكتوبة completed لكن إنتِ ما خلصتي دوراتك
                if task_status == "completed" and max_turns > 0 and your_turn < max_turns:
                    task_status = "progress"
        except Exception as e:
            app.logger.exception("AI get: failed to load task status: %s", e)

        return jsonify({
            "messages": messages,
            "currentTurn": your_turn,
            "taskStatus": task_status
        }), 200
        
    except Exception as e:
        print("🔥 AI get error:", e)
        return jsonify({"error": "Server error"}), 500


# ==================================================
# Task Update H-H
# ==================================================
def _compute_hh_turns_for_examiner(msgs, examiner_id, examiner_ids):
    """
    يحسب عدد الـ turns لممتحِن واحد في محادثة Human-Human.

    turn واحد = (block من self) + (block من peer) أو العكس.
    البلوك = مجموعة رسائل متتالية من نفس الطرف.
    """
    speaker_seq = []

    for m in msgs:
        if not isinstance(m, dict):
            continue

        sender = m.get("examiner_id") or m.get("sender_id")
        if sender not in examiner_ids:
            continue

        if sender == examiner_id:
            speaker_seq.append("self")
        else:
            speaker_seq.append("peer")

    if not speaker_seq:
        return 0

    # ندمج البلوكات المتتالية المتشابهة
    runs = []
    last = None
    for s in speaker_seq:
        if s != last:
            runs.append(s)
            last = s

    # كل بلوكين متتاليين (self+peer أو peer+self) = 1 turn مكتمل
    turns = len(runs) // 2
    return turns

def _update_hh_task_status_if_completed(task_id):
    """
    يشيّك إذا كل الـ examiners في محادثة Human-Human
    خلصوا عدد الـ turns المطلوب بناءً على تعريفك للـ turn:

    turn واحد = (block من رسائل self) + (block من رسائل peer) أو العكس،
    بغض النظر عن عدد الرسائل داخل كل block.
    """
    try:
        task_ref = db.collection("tasks").document(task_id)
        task_doc = task_ref.get()
        if not task_doc.exists:
            return

        task_data = task_doc.to_dict()

        # نتأكد أنها مهمة Human-Human
        if task_data.get("conversation_type") != "human-human":
            return

        max_turns = int(task_data.get("number_of_turns", 0) or 0)
        if max_turns <= 0:
            return

        examiner_ids = task_data.get("examiner_ids") or []

        # لو ما فيه examiner_ids (حالات قديمة) نجمعهم من الرسائل
        conv_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
        raw = conv_ref.get() or {}
        if isinstance(raw, dict):
            msgs = list(raw.values())
        elif isinstance(raw, list):
            msgs = raw
        else:
            msgs = []

        if not examiner_ids:
            examiner_ids = list({
                m.get("examiner_id")
                for m in msgs
                if isinstance(m, dict) and m.get("examiner_id")
            })

        if not examiner_ids or not msgs:
            return

        # نرتب الرسائل زمنيًا
        msgs.sort(key=lambda m: m.get("created_at", ""))

        # نحسب عدد الـ turns لكل ممتحِن
        turns_per_examiner = {}
        for ex_id in examiner_ids:
            t = _compute_hh_turns_for_examiner(msgs, ex_id, examiner_ids)
            if max_turns > 0:
                t = min(t, max_turns)
            turns_per_examiner[ex_id] = t

        # نكمّل التاسك فقط لو كلهم وصلوا max_turns
        completed = all(turns_per_examiner.get(e, 0) >= max_turns for e in examiner_ids)

        if completed and task_data.get("status") != "completed":
            task_ref.update({"status": "completed"})

    except Exception as e:
        app.logger.exception("Failed to update HH task status: %s", e)
# ==================================================
# Task Update Ai
# ==================================================

def _update_ai_task_status_if_completed(task_id):
    """
    يشيّك إذا كل الـ examiners وصلوا لعدد الـ turns المطلوب
    (محادثة Human-AI) ولو نعم يحدّث حالة التاسك إلى completed.
    """
    try:
        task_ref = db.collection("tasks").document(task_id)
        task_doc = task_ref.get()
        if not task_doc.exists:
            return

        task_data = task_doc.to_dict()

        # نتأكد أنه Human-AI
        if task_data.get("conversation_type") != "human-ai":
            return

        max_turns = int(task_data.get("number_of_turns", 0) or 0)
        if max_turns <= 0:
            return

        examiner_ids = task_data.get("examiner_ids") or []
        if not examiner_ids:
            # لو ما فيه examiner_ids لأي سبب، نجمعهم من الرسائل
            conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
            raw = conv_ref.get() or {}
            if isinstance(raw, dict):
                msgs = raw.values()
            elif isinstance(raw, list):
                msgs = raw
            else:
                msgs = []

            examiner_ids = list({
                m.get("examiner_id")
                for m in msgs
                if isinstance(m, dict) and m.get("examiner_id")
            })

        if not examiner_ids:
            return

        # نقرأ كل رسائل هذه المحادثة
        conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
        raw = conv_ref.get() or {}
        if isinstance(raw, dict):
            msgs = raw.values()
        elif isinstance(raw, list):
            msgs = raw
        else:
            msgs = []

        # نحسب كم رسالة كتب كل examiner (sender_type == "Ex")
        counts = {ex_id: 0 for ex_id in examiner_ids}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("sender_type") != "Ex":
                continue
            ex_id = m.get("examiner_id")
            if ex_id in counts:
                counts[ex_id] += 1

        # لو كل واحد وصل على الأقل max_turns → نكمّل التاسك
        completed = all(counts.get(e, 0) >= max_turns for e in examiner_ids)

        if completed and task_data.get("status") != "completed":
            task_ref.update({"status": "completed"})

    except Exception as e:
        app.logger.exception("Failed to update AI task status: %s", e)



# ==================================================
# 🔹 Human ↔ Human Conversation APIs (Realtime DB)
# ==================================================

# 1) جلب كل رسائل التاسك من الـ Realtime DB
@app.route("/api/hh/messages", methods=["GET"])
def api_hh_get_messages():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "Missing taskId"}), 400

    uid = session.get("uid")

    try:
        ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
        raw = ref.get() or {}

        # نحولها لقائمة
        if isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []

        # نرتب الرسائل بالوقت
        rows.sort(key=lambda m: m.get("created_at", ""))

        messages = []
        speaker_sequence = []   # "you" أو "peer" بالترتيب الزمني

        for m in rows:
            if not isinstance(m, dict):
                continue

            sender_id = m.get("examiner_id") or m.get("sender_id")
            sender_name = (m.get("sender_name") or "User").strip() or "User"
            text = m.get("message", "")

            if not sender_id:
                continue

            if sender_id == uid:
                side = "you"
                speaker_sequence.append("you")
            else:
                side = "peer"
                speaker_sequence.append("peer")

            initial = (sender_name[0].upper() if sender_name else "U")

            messages.append({
                "text": text,
                "side": side,
                "authorInitial": initial,
            })

        # ======== حساب عدد الـ turns من وجهة نظرك ========
        # نحول sequence إلى blocks متتالية مختلفة
        runs = []
        last = None
        for s in speaker_sequence:
            if s != last:
                runs.append(s)
                last = s

        # كل (you + peer) أو (peer + you) = turn واحد
        your_turn = len(runs) // 2

        task_status = "pending"
        max_turns = 0

        try:
            task_doc = db.collection("tasks").document(task_id).get()
            if task_doc.exists:
                tdata = task_doc.to_dict()
                task_status = tdata.get("status", "pending")
                max_turns = int(tdata.get("number_of_turns", 0) or 0)

                if max_turns > 0:
                    your_turn = min(your_turn, max_turns)

                # لو التاسك مكتوب completed بس لسه ما خلصتي كل التيرنز → نخليها progress
                if task_status == "completed" and max_turns > 0 and your_turn < max_turns:
                    task_status = "progress"
        except Exception as e:
            app.logger.exception("HH get: failed to load task status: %s", e)

        return jsonify({
            "messages": messages,
            "currentTurn": your_turn,
            "taskStatus": task_status
        }), 200

    except Exception as e:
        print("🔥 HH get error:", e)
        return jsonify({"error": "Server error"}), 500

# 2) حفظ رسالة جديدة في الـ Realtime DB
# 2) حفظ رسالة جديدة في الـ Realtime DB
@app.route("/api/hh/send", methods=["POST"])
def api_hh_send():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}

    task_id = data.get("task_id") or data.get("taskId")
    message = (data.get("message") or data.get("text") or "").strip()

    if not task_id or not message:
        return jsonify({"error": "Missing taskId or message"}), 400

    sender_id = session.get("uid")
    if not sender_id:
        return jsonify({"error": "Missing uid in session"}), 401

    sender_doc = db.collection("users").document(sender_id).get()
    sender_name = "User"
    if sender_doc.exists:
        prof = sender_doc.to_dict().get("profile", {})
        sender_name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip() or "User"

    ref = rtdb.reference(f"hh_conversations/{task_id}/messages")

    # 🧠 نجيب الرسائل الموجودة
    existing = ref.get() or {}
    rows = []
    if isinstance(existing, dict):
        rows = list(existing.values())
    elif isinstance(existing, list):
        rows = existing

    # ==============================
    # 🔒 منع إرسال رسالتين ورا بعض
    # ==============================
    if rows:
        # نرتّب الرسائل بالوقت
        try:
            rows_sorted = sorted(rows, key=lambda r: r.get("created_at", ""))
        except Exception:
            rows_sorted = rows

        last_msg = rows_sorted[-1]

        # نركز على رسائل الـ examiners فقط
        if (
            isinstance(last_msg, dict)
            and last_msg.get("sender_type") == "Ex"
            and last_msg.get("examiner_id") == sender_id
        ):
            # نفس الشخص أرسل آخر رسالة → لازم ينتظر الثاني
            return jsonify({
                "error": "WAIT_FOR_PEER",
                "message": "You must wait for the other examiner to reply before sending another message."
            }), 400

    # نحسب turn_number الخاص بهذا الـ examiner فقط
    count_for_this_ex = 0
    for row in rows:
        if isinstance(row, dict) and row.get("examiner_id") == sender_id:
            count_for_this_ex += 1

    next_turn_number = count_for_this_ex + 1
    turn_id = str(uuid.uuid4())

    try:
        ref.push({
            "turn_id": turn_id,
            "task_id": task_id,
            "turn_number": next_turn_number,
            "sender_type": "Ex",
            "examiner_id": sender_id,
            "message": message,
            "sender_name": sender_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })

        _update_hh_task_status_if_completed(task_id)

        return jsonify({"success": True, "message": "Message saved"}), 200

    except Exception as e:
        print("🔥 HH send error:", e)
        return jsonify({"error": "Server error"}), 500



@app.route("/api/hh/messages_owner", methods=["GET"])
def api_hh_messages_owner():
    """
    عرض محادثة Human ↔ Human للـ Owner من مسار:
    hh_conversations/{taskId}/messages/{pushId}
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "taskId is required"}), 400

    try:
        messages_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
        raw = messages_ref.get() or {}

        # نحول النودات إلى list ونرتبها حسب turn_number ثم created_at
        all_msgs = []
        for key, val in (raw or {}).items():
            if not isinstance(val, dict):
                continue
            val["_key"] = key
            all_msgs.append(val)

        def _as_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        all_msgs.sort(
            key=lambda m: (
                _as_int(m.get("turn_number", 0)),
                m.get("created_at") or ""
            )
        )

        # نحدد examiners عشان نوزعهم يسار/يمين
        examiner_side = {}
        side_order = ["left", "right"]

        def get_side_for_examiner(ex_id):
            if not ex_id:
                return "left"
            if ex_id not in examiner_side:
                # أول واحد يصير left، الثاني right
                examiner_side[ex_id] = side_order[len(examiner_side) % 2]
            return examiner_side[ex_id]

        msgs = []
        max_turn = 0

        for m in all_msgs:
            text = m.get("message") or ""
            if not text:
                continue

            turn_number = _as_int(m.get("turn_number", 0))
            if turn_number > max_turn:
                max_turn = turn_number

            examiner_id = m.get("examiner_id")
            sender_name = m.get("sender_name") or "Examiner"

            side = get_side_for_examiner(examiner_id)

            msgs.append({
                "text": text,
                "side": side,  # left / right
                "author": sender_name,
                "authorLabel": sender_name,
                "turnIndex": turn_number,
            })

        return jsonify({
            "messages": msgs,
            "currentTurn": max_turn,
            "isComplete": False,   # ما عندنا فلاغ واضح في السكيمة الحالية
        }), 200

    except Exception as e:
        app.logger.exception("Error in api_hh_messages_owner: %s", e)
        return jsonify({"error": "Server error while loading HH conversation"}), 500

@app.route("/api/llm/messages_owner", methods=["GET"])
def api_llm_messages_owner():
    """
    عرض محادثة Human ↔ LLM للـ Owner من مسار:
    llm_conversations/{taskId}/messages/{pushId}
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "taskId is required"}), 400

    try:
        messages_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
        raw = messages_ref.get() or {}

        all_msgs = []
        for key, val in (raw or {}).items():
            if not isinstance(val, dict):
                continue
            val["_key"] = key
            all_msgs.append(val)

        def _as_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        all_msgs.sort(
            key=lambda m: (
                _as_int(m.get("turn_number", 0)),
                m.get("created_at") or ""
            )
        )

        # نفترض إن الـ human عنده examiner_id، والـ LLM غالبًا بدون examiner_id
        examiner_side = {}

        def get_side(msg):
            st = (msg.get("sender_type") or "").lower()
            ex_id = msg.get("examiner_id")

            # لو رسالة من الـ LLM
            if st in ("llm", "ai", "assistant", "model") or (not ex_id):
                return "right"

            # البشري
            if ex_id not in examiner_side:
                examiner_side[ex_id] = "left"
            return examiner_side[ex_id]

        msgs = []
        max_turn = 0

        for m in all_msgs:
            text = m.get("message") or ""
            if not text:
                continue

            turn_number = _as_int(m.get("turn_number", 0))
            if turn_number > max_turn:
                max_turn = turn_number

            sender_name = m.get("sender_name") or "Speaker"

            side = get_side(m)

            msgs.append({
                "text": text,
                "side": side,  # left = human, right = LLM
                "author": sender_name,
                "authorLabel": sender_name,
                "turnIndex": turn_number,
            })


        return jsonify({
            "messages": msgs,
            "currentTurn": max_turn,
            "isComplete": False,
        }), 200

    except Exception as e:
        app.logger.exception("Error in api_llm_messages_owner: %s", e)
        return jsonify({"error": "Server error while loading LLM conversation"}), 500
    
    
  
# ===================================================================


@app.route("/api/project/<project_id>/dataset", methods=["GET"])
def get_project_dataset(project_id):
    """
يجيب كل مقالات الـ dataset حق مشروع معين
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    # 1️⃣ نجيب بيانات المشروع من Firestore
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()
    
    # 2️⃣ نتحقق من الصلاحية (Owner أو Examiner مقبول)
    is_owner = proj_data.get("owner_id") == uid
    
    is_examiner = False
    if not is_owner:
        # نشيك إذا هو examiner مقبول
        inv_docs = list(
            db.collection("invitations")
            .where("project_id", "==", project_id)
            .where("examiner_id", "==", uid)
            .where("status", "==", "accepted")
            .limit(1)
            .stream()
        )
        is_examiner = len(inv_docs) > 0

    if not is_owner and not is_examiner:
        return jsonify({"error": "Forbidden"}), 403

    # 3️⃣ نجيب dataset_id
    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found for this project"}), 404

    # 4️⃣ نسحب كل المقالات من Realtime Database
    try:
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get()
        
        if not snapshot:
            return jsonify({"articles": [], "total": 0}), 200

        articles = []
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue
                
            payload = article_data.get("payload", {})
            
            # نستخرج النص (fallback)
            title = (payload.get("title") or 
                    payload.get("Title") or 
                    payload.get("headline") or "")
            
            content = (payload.get("Article") or 
                      payload.get("article") or 
                      payload.get("content") or 
                      payload.get("text") or "")
            
            articles.append({
                "id": push_id,
                "title": title,
                "content": content,
                "full_text": f"{title}. {content}" if title else content
            })

        return jsonify({
            "articles": articles,
            "total": len(articles),
            "dataset_id": dataset_id
        }), 200

    except Exception as e:
        app.logger.exception("Failed to fetch dataset: %s", e)
        return jsonify({"error": "Failed to fetch dataset"}), 500

@app.route("/api/project/<project_id>/analyze_all", methods=["POST"])
def analyze_all_articles(project_id):
    """
    يحلل كل مقالات المشروع بالنموذج ويحفظ النتائج
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    # نتحقق إن المستخدم هو الـ Owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Only project owner can run batch analysis"}), 403
    
    data = request.get_json(silent=True) or {}
    selected_model = (data.get("model") or "logistic").lower()



    # نسحب الـ dataset
    dataset_id = proj_doc.to_dict().get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found"}), 404

    try:
        # نجيب كل المقالات
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get()

        if not snapshot:
            return jsonify({"error": "Dataset is empty"}), 404

        results = []
        human_count = 0
        ai_count = 0

        # نحلل كل مقالة
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue

            payload = article_data.get("payload", {})
            
            title = (payload.get("title") or 
                    payload.get("Title") or 
                    payload.get("headline") or "")
            
            content = (payload.get("Article") or 
                      payload.get("article") or 
                      payload.get("content") or "")
            
            full_text = f"{title}. {content}" if title else content

            # تحليل بالنموذج
            chunks = split_into_3_chunks(full_text)
            
            human_scores = []
            ai_scores = []
            
            for chunk in chunks:
                if selected_model == "rnn":
                    probabilities = rnn_predict_proba([chunk])[0]   # لاحظي [chunk]
                else:
                    probabilities = news_pipeline.predict_proba([chunk])[0]

           #  نحولها إلى float عادي
                human_scores.append(float(probabilities[0]))
                ai_scores.append(float(probabilities[1]))


            
            # المتوسط
            final_human = sum(human_scores) / len(human_scores)
            final_ai = sum(ai_scores) / len(ai_scores)
          
            #4 النتيجة النهائية
            prediction = "AI" if final_ai > final_human else "Human"
            
            if prediction == "Human":
                human_count += 1
            else:
                ai_count += 1

            results.append({
                "confidence": round(max(final_human, final_ai) * 100, 2),
                "confidence": round(max(final_human, final_ai) * 100, 2),
    "article_id": push_id,
    "title": title[:100] if title else "",
    "content": full_text[:500] if full_text else "",
    "prediction": prediction,
    "human_percentage": round(final_human * 100, 2),
    "ai_percentage": round(final_ai * 100, 2),
    "chunks": [
        {
            "label": f"F{i+1}",
            "human": round(float(human_scores[i]) * 100, 2),
            "ai": round(float(ai_scores[i]) * 100, 2)
        }
        for i in range(len(chunks))
    ]
})

        # نحفظ النتائج في Firestore
        analysis_doc = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_type": selected_model,
            "total_articles": len(results),
            "human_count": human_count,
            "ai_count": ai_count,
            "human_percentage": round((human_count / len(results)) * 100, 2),
            "ai_percentage": round((ai_count / len(results)) * 100, 2),
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
            "analyzed_by": uid
        }


        # نرتب النتائج من الأقل ثقة للأعلى
        results.sort(key=lambda x: x["confidence"])


        # نرتب النتائج من الأقل ثقة للأعلى
        results.sort(key=lambda x: x["confidence"])

        db.collection("project_analysis").document(project_id).set(analysis_doc)

        # نحفظ النتائج التفصيلية في Realtime DB
        results_ref = rtdb.reference(f"analysis_results/{project_id}/{selected_model}")
        results_ref.set({
            "summary": analysis_doc,
            "details": results
        })


        return jsonify({
            "message": "Analysis complete",
            "summary": analysis_doc,
            "total_analyzed": len(results)
        }), 200

    except Exception as e:
        app.logger.exception("Batch analysis failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500


# ═══════════════════════════════════════════════════════════════
# 🔬 Model Selection Task Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/task/<task_id>/model-selection")
def model_selection_task_page(task_id):
    """صفحة Model Selection للـ Examiner"""
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    
    # نجيب معلومات Task
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        abort(404)
    
    task_data = task_doc.to_dict()
    
    # نتأكد إنه Model Selection Task
    if task_data.get("task_type") != "model_selection":
        abort(403)
    
    # نتأكد إن الـ Examiner مسند له
    if uid not in task_data.get("examiner_ids", []):
        abort(403)
    
    # نجيب اسم المستخدم
    user_doc = db.collection("users").document(uid).get()
    first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
    last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
    user_name = f"{first_name} {last_name}".strip() or "User"
    
    project_id = task_data.get("project_ID")
    
    return render_template(
        "ModelSelectionTask.html",
        user_name=user_name,
        task_id=task_id,
        project_id=project_id,
        task_name=task_data.get("task_name", "Model Selection")
    )


@app.route("/api/task/<task_id>/run_model", methods=["POST"])
def api_run_model(task_id):
    """يشغل Model على Dataset"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    
    # نجيب Task
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404
    
    task_data = task_doc.to_dict()
    
    # نتأكد من الصلاحية
    if uid not in task_data.get("examiner_ids", []):
        return jsonify({"error": "Forbidden"}), 403
    
    # نجيب المشروع
    project_id = task_data.get("project_ID")
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    dataset_id = proj_doc.to_dict().get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset"}), 404
    
    # نقرأ Model المطلوب
    data = request.get_json() or {}
    model_type = (data.get("model") or "logistic").lower()
    
    try:
        # نسحب Dataset
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get()
        
        if not snapshot:
            return jsonify({"error": "Dataset is empty"}), 404
        
        results = []
        human_count = 0
        ai_count = 0
        
        # نحلل كل مقالة
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue
            
            payload = article_data.get("payload", {})
            
            title = (payload.get("title") or 
                    payload.get("Title") or "")
            
            content = (payload.get("Article") or 
                      payload.get("article") or "")
            
            full_text = f"{title}. {content}" if title else content
            
            chunks = split_into_3_chunks(full_text)
            
            human_scores = []
            ai_scores = []
            chunk_details = []
            
            for i, chunk in enumerate(chunks):
                if model_type == "rnn":
                    probabilities = rnn_predict_proba([chunk])[0]
                else:
                    probabilities = news_pipeline.predict_proba([chunk])[0]
                
                h_score = float(probabilities[0])
                a_score = float(probabilities[1])
                
                human_scores.append(h_score)
                ai_scores.append(a_score)
                
                chunk_details.append({
                    "label": f"F{i+1}",
                    "human": round(h_score * 100, 2),
                    "ai": round(a_score * 100, 2)
                })
            
            final_human = sum(human_scores) / len(human_scores)
            final_ai = sum(ai_scores) / len(ai_scores)
            
            prediction = "AI" if final_ai > final_human else "Human"
            
            if prediction == "Human":
                human_count += 1
            else:
                ai_count += 1
            
            results.append({
                "confidence": round(max(final_human, final_ai) * 100, 2),
                "confidence": round(max(final_human, final_ai) * 100, 2),
                "article_id": push_id,
                "title": title[:100] if title else "Untitled",
                "content": content[:500] if content else "",  # ✅ أول 500 حرف
                "prediction": prediction,
                "human_percentage": round(final_human * 100, 2),
                "ai_percentage": round(final_ai * 100, 2),
                "chunks": chunk_details  # ✅ تفاصيل الـ Chunks
            })
        
        # ملخص النتائج
        summary = {
            "model_type": model_type,
            "total_articles": len(results),
            "human_count": human_count,
            "ai_count": ai_count,
            "human_percentage": round((human_count / len(results)) * 100, 2) if results else 0,
            "ai_percentage": round((ai_count / len(results)) * 100, 2) if results else 0
        }
        
        return jsonify({
            "summary": summary,
            "sample_results": results[:10],  # أول 10
            "all_results": results  # ✅ كل النتائج مع التفاصيل
        }), 200
        
    except Exception as e:
        app.logger.exception("Model execution failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500


@app.route("/api/task/<task_id>/select_model", methods=["POST"])
def api_select_model(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict()

    if uid not in task_data.get("examiner_ids", []):
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json() or {}
    selected_model = (data.get("model") or "logistic").lower()

    if selected_model not in ["logistic", "rnn"]:
        return jsonify({"error": "Invalid model"}), 400

    # نحفظ الاختيار في Firestore
    task_ref.update({
        "selected_model": selected_model,
        "selected_by": uid,
        "selected_at": datetime.utcnow().isoformat() + "Z",
        "status": "completed"
    })

    # نشغل التحليل ونحفظ في RTDB
    try:
        project_id = task_data.get("project_ID")
        proj_doc = db.collection("projects").document(project_id).get()
        dataset_id = proj_doc.to_dict().get("dataset_id") if proj_doc.exists else None

        if dataset_id:
            ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
            snapshot = ref.get() or {}

            results = []
            human_count = 0
            ai_count = 0

            for push_id, article_data in snapshot.items():
                if not isinstance(article_data, dict):
                    continue

                payload = article_data.get("payload", {})
                title = payload.get("title") or payload.get("Title") or ""
                content = payload.get("Article") or payload.get("article") or ""
                full_text = f"{title}. {content}" if title else content

                chunks = split_into_3_chunks(full_text)
                human_scores = []
                ai_scores = []
                chunk_details = []

                for i, chunk in enumerate(chunks):
                    if selected_model == "rnn":
                        probabilities = rnn_predict_proba([chunk])[0]
                    else:
                        probabilities = news_pipeline.predict_proba([chunk])[0]

                    h = float(probabilities[0])
                    a = float(probabilities[1])
                    human_scores.append(h)
                    ai_scores.append(a)
                    chunk_details.append({
                        "label": f"F{i+1}",
                        "human": round(h * 100, 2),
                        "ai": round(a * 100, 2)
                    })

                final_human = sum(human_scores) / len(human_scores)
                final_ai = sum(ai_scores) / len(ai_scores)
                prediction = "AI" if final_ai > final_human else "Human"

                if prediction == "Human":
                    human_count += 1
                else:
                    ai_count += 1

                results.append({
                    "confidence": round(max(final_human, final_ai) * 100, 2),
                    "confidence": round(max(final_human, final_ai) * 100, 2),
                    "article_id": push_id,
                    "title": title[:100],
                    "content": content[:500],
                    "prediction": prediction,
                    "human_percentage": round(final_human * 100, 2),
                    "ai_percentage": round(final_ai * 100, 2),
                    "chunks": chunk_details
                })

            # نحفظ في RTDB - نظف project_id من الأحرف الممنوعة
            safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
            results_ref = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}")
            results_ref.set({
                "summary": {
                    "model_type": selected_model,
                    "total_articles": len(results),
                    "human_count": human_count,
                    "ai_count": ai_count,
                },
                "details": results
            })
            app.logger.info("=== Analysis saved to RTDB: analysis_results/%s/%s ===", safe_pid, selected_model)

    except Exception as e:
        app.logger.exception("Failed to run analysis after model selection: %s", e)

    return jsonify({"message": "Model selected successfully"}), 200

# 📝 Examiner Feedback APIs
# ═══════════════════════════════════════════════════════════════

@app.route('/api/article/<article_id>/feedback', methods=['POST'])
def submit_article_feedback(article_id):
    """Submit examiner feedback"""
    try:
        if not session.get('idToken'):
            return jsonify({"error": "Unauthorized"}), 401
        
        user_id = session.get('uid')
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        user_role = user_doc.to_dict().get('role')
        if user_role != 'examiner':
            return jsonify({"error": "Only examiners can submit feedback"}), 403
        
        data = request.get_json()
        label = data.get('label')
        explanation = data.get('explanation', '').strip()
        
        if not label or label not in ['Human', 'AI']:
            return jsonify({"error": "Invalid label"}), 400
        
        if not explanation:
            return jsonify({"error": "Explanation is required"}), 400
        
        # Get examiner name
        first_name = user_doc.to_dict().get('profile', {}).get('firstName', '')
        last_name = user_doc.to_dict().get('profile', {}).get('lastName', '')
        examiner_name = f"{first_name} {last_name}".strip() or "Unknown"
        
        # Find article
        uploaded_news_ref = rtdb.reference('datasets/uploaded_news')
        article_found = False
        article_path = None
        
        all_datasets = uploaded_news_ref.get() or {}
        for dataset_id, dataset_content in all_datasets.items():
            if isinstance(dataset_content, dict):
                for push_id, article_data in dataset_content.items():
                    if push_id == article_id:
                        article_found = True
                        article_path = f'datasets/uploaded_news/{dataset_id}/{article_id}'
                        break
            if article_found:
                break
        
        if not article_found:
            return jsonify({"error": "Article not found"}), 404
        
        feedback_data = {
            'label': label,
            'explanation': explanation,
            'examiner_name': examiner_name,
            'submitted_at': datetime.utcnow().isoformat() + "Z"
        }
        
        feedback_ref = rtdb.reference(f'{article_path}/examiner_feedbacks/{user_id}')
        feedback_ref.set(feedback_data)
        
        return jsonify({"success": True, "message": "Feedback submitted", "feedback": feedback_data}), 200
        
    except Exception as e:
        app.logger.exception("Error submitting feedback: %s", e)
        return jsonify({"error": "Failed to submit feedback"}), 500


@app.route('/api/article/<article_id>/feedbacks', methods=['GET'])
def get_article_feedbacks(article_id):
    """Get feedbacks: Owner sees ALL, Examiner sees ONLY theirs"""
    try:
        if not session.get('idToken'):
            return jsonify({"error": "Unauthorized"}), 401
        
        user_id = session.get('uid')
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        user_role = user_doc.to_dict().get('role')
        
        # Find article
        uploaded_news_ref = rtdb.reference('datasets/uploaded_news')
        article_found = False
        article_path = None
        
        all_datasets = uploaded_news_ref.get() or {}
        for dataset_id, dataset_content in all_datasets.items():
            if isinstance(dataset_content, dict):
                for push_id, article_data in dataset_content.items():
                    if push_id == article_id:
                        article_found = True
                        article_path = f'datasets/uploaded_news/{dataset_id}/{article_id}'
                        break
            if article_found:
                break
        
        if not article_found:
            return jsonify({"error": "Article not found"}), 404
        
        feedbacks_ref = rtdb.reference(f'{article_path}/examiner_feedbacks')
        all_feedbacks = feedbacks_ref.get() or {}
        
        # Filter based on role
        if user_role == 'examiner':
            if user_id in all_feedbacks:
                return jsonify({"feedbacks": {user_id: all_feedbacks[user_id]}, "has_feedback": True}), 200
            else:
                return jsonify({"feedbacks": {}, "has_feedback": False}), 200
        
        elif user_role == 'project_owner':
            return jsonify({"feedbacks": all_feedbacks, "total_count": len(all_feedbacks)}), 200
        
        else:
            return jsonify({"error": "Invalid role"}), 403
        
    except Exception as e:
        app.logger.exception("Error getting feedbacks: %s", e)
        return jsonify({"error": "Failed to get feedbacks"}), 500
    
    
# ═══════════════════════════════════════════════════════════════
# 📝 FeedBack Task Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/task/<task_id>/feedback")
def feedback_task_page(task_id):
    """صفحة FeedBack للـ Examiner"""
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    
    # نجيب معلومات Task
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        abort(404)
    
    task_data = task_doc.to_dict()
    
    # نتأكد إنه FeedBack Task
    if task_data.get("task_type") != "labeling":
        abort(403)
    
    # نتأكد إن الـ Examiner مسند له
    if uid not in task_data.get("examiner_ids", []):
        abort(403)
    
    # نجيب اسم المستخدم
    user_doc = db.collection("users").document(uid).get()
    first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
    last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
    user_name = f"{first_name} {last_name}".strip() or "User"
    
    project_id = task_data.get("project_ID")
    
    project_id = task_data.get("project_ID")
    
    # ✅ نجيب dataset_id
    proj_doc = db.collection("projects").document(project_id).get()
    dataset_id = proj_doc.to_dict().get("dataset_id", "") if proj_doc.exists else ""
    
    return render_template(
        "feedbacktask.html",
        user_name=user_name,
        task_id=task_id,
        project_id=project_id,
        dataset_id=dataset_id,  # ✅ نمرره للـ Template
        task_name=task_data.get("task_name", "Feedback Task")
    )
    

@app.route("/api/task/<task_id>/articles", methods=["GET"])
def api_get_task_articles(task_id):
    try:
        if not session.get("idToken"):
            return jsonify({"error": "Unauthorized"}), 401

        uid = session.get("uid")

        task_doc = db.collection("tasks").document(task_id).get()
        if not task_doc.exists:
            return jsonify({"error": "Task not found"}), 404

        task_data = task_doc.to_dict()
        project_id = task_data.get("project_ID")

        app.logger.info("=== DEBUG === project_id: %s", project_id)
        all_tasks = list(db.collection("tasks").where("project_ID", "==", project_id).stream())
        for t in all_tasks:
            d = t.to_dict()
            app.logger.info("=== DEBUG === task: type=%s status=%s", d.get("task_type"), d.get("status"))

        # نشيك إذا في model_selection task مكتمل
        model_selection_tasks = list(
            db.collection("tasks")
            .where("project_ID", "==", project_id)
            .where("task_type", "==", "model_selection")
            .where("status", "==", "completed")
            .limit(1)
            .stream()
        )

        if not model_selection_tasks:
            return jsonify({
                "waiting": True,
                "message": "Waiting for model selection task to be completed"
            }), 200

        ms_task = model_selection_tasks[0].to_dict()
        selected_model = ms_task.get("selected_model", "logistic")

        app.logger.info("=== DEBUG === selected_model: %s", selected_model)

        # نجيب dataset_id
        proj_doc = db.collection("projects").document(project_id).get()
        if not proj_doc.exists:
            return jsonify({"error": "Project not found"}), 404

        dataset_id = proj_doc.to_dict().get("dataset_id")
        if not dataset_id:
            return jsonify({"error": "No dataset found"}), 404

        # نجيب النتائج من RTDB
        safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
        results_ref = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}")
        results_data = results_ref.get()

        app.logger.info("=== DEBUG === results_data exists: %s", results_data is not None)

        if not results_data:
            return jsonify({
                "waiting": True,
                "message": "Analysis not ready yet, please wait"
            }), 200

        details = results_data.get("details", [])
        summary = results_data.get("summary", {})

        # نجيب الـ feedbacks الموجودة
        articles_with_feedback = []
        for article in details:
            article_id = article.get("article_id")
            feedback = None

            try:
                feedback_ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}/{article_id}/feedback")
                feedback = feedback_ref.get()
            except Exception:
                pass

            articles_with_feedback.append({
                "article_id": article_id,
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "prediction": article.get("prediction", ""),
                "human_percentage": article.get("human_percentage", 0),
                "ai_percentage": article.get("ai_percentage", 0),
                "confidence": article.get("confidence"),
                "chunks": article.get("chunks", []),
                "has_feedback": feedback is not None,
                "feedback": feedback
            })

        return jsonify({
            "articles": articles_with_feedback,
            "summary": summary,
            "selected_model": selected_model,
            "total": len(articles_with_feedback)
        }), 200

    except Exception as e:
        app.logger.exception("Unexpected error in api_get_task_articles: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/article/<article_id>/submit_feedback", methods=["POST"])
def api_submit_article_feedback(article_id):
    """يحفظ Feedback من Examiner"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    
    data = request.get_json() or {}
    label = data.get("label")
    explanation = data.get("explanation", "").strip()
    dataset_id = data.get("dataset_id")
    
    if not label or label not in ["Human", "AI"]:
        return jsonify({"error": "Invalid label"}), 400
    
    if not explanation:
        return jsonify({"error": "Explanation is required"}), 400
    
    if not dataset_id:
        return jsonify({"error": "Dataset ID is required"}), 400
    
    try:
        # نشيك لو فيه feedback قبل
        feedback_ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}/{article_id}/feedback")
        existing = feedback_ref.get()
        
        if existing:
            return jsonify({"error": "Feedback already exists for this article"}), 400
        
        # نجيب اسم الـ Examiner
        user_doc = db.collection("users").document(uid).get()
        first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
        last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
        examiner_name = f"{first_name} {last_name}".strip() or "Examiner"
        
        # نحفظ الـ Feedback
        feedback_data = {
            "examiner_uid": uid,
            "examiner_name": examiner_name,
            "label": label,
            "explanation": explanation,
            "submitted_at": datetime.utcnow().isoformat() + "Z"
        }
        
        feedback_ref.set(feedback_data)
        
        return jsonify({"message": "Feedback submitted successfully"}), 200
        
    except Exception as e:
        app.logger.exception("Failed to submit feedback: %s", e)
        return jsonify({"error": "Failed to submit feedback"}), 500
    
if __name__ == "__main__":
 app.run(debug=True)