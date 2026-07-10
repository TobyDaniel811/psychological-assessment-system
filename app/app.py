"""
app/app.py
----------------------------------------------------------------------
Main Flask application.

Routes:
  - Home
  - Register / Login / Logout
  - Survey (GET renders form, POST processes + predicts + saves)
  - Result (shows prediction + plain-language explanation)
  - History
  - Admin Dashboard
----------------------------------------------------------------------
"""

import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
import joblib

from app.database import init_db, get_connection
from app.models import User
from app.predictor import predict_condition, ordinal_maps, nominal_cols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMINAL_OPTIONS = joblib.load(os.path.join(BASE_DIR, "model", "nominal_options.pkl"))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = "dev-secret-key-change-this-before-deployment"

# --- Flask-Login setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"          # where to redirect if @login_required fails
login_manager.login_message = "Please log in to access that page."


@login_manager.user_loader
def load_user(user_id):
    """Flask-Login calls this on every request to reload the user object
    from the id stored in the session cookie."""
    return User.get_by_id(user_id)


def admin_required(view_func):
    """
    Custom decorator that builds on top of @login_required.
    Must be used TOGETHER with @login_required, and placed BELOW it,
    e.g.:

        @app.route("/admin")
        @login_required
        @admin_required
        def admin_dashboard():
            ...

    If the logged-in user's role is not 'admin', they are redirected
    home with a flash message instead of seeing the page or getting
    a raw 403 error.
    """
    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("You do not have permission to view that page.", "error")
            return redirect(url_for("home"))
        return view_func(*args, **kwargs)

    return wrapped


# Ensures the database tables exist whether the app is started via
# `python app/app.py`, `flask run`, or a production WSGI server -- not
# only through the __main__ block below.
init_db()


# ----------------------------------------------------------------------
# CONDITION EXPLANATIONS
# Plain-language descriptions shown on the result page.
# Each entry has: icon, colour, summary, what_it_means, signs, what_to_do.
# Written in behavioural assessment language -- NOT clinical diagnosis.
# ----------------------------------------------------------------------
CONDITION_INFO = {
    "Mood Disorders": {
        "icon": "🌧",
        "colour": "#5b8dee",
        "summary": "Your responses reflect patterns commonly associated with mood disturbances.",
        "what_it_means": (
            "Mood disorders cover a range of conditions that affect how a person feels "
            "emotionally over a sustained period. Your responses suggest you may be "
            "experiencing persistent low mood, reduced motivation, or emotional "
            "instability that is interfering with your day-to-day functioning. "
            "These patterns are very common and highly responsive to the right support."
        ),
        "signs": [
            "Persistent feelings of sadness or emptiness",
            "Loss of interest in things that used to feel enjoyable",
            "Fatigue or low energy even without physical exertion",
            "Difficulty finding motivation to start or finish tasks",
            "Feelings of hopelessness or worthlessness",
        ],
        "what_to_do": [
            "Speak with a counsellor, therapist, or doctor as a first step — mood disorders respond well to professional support.",
            "Try to maintain a regular daily routine, including consistent sleep and mealtimes.",
            "Reach out to someone you trust — isolation tends to make low mood worse.",
            "Gentle physical activity, even short walks, has a measurable positive effect on mood.",
            "Avoid making major life decisions during a period of low mood where possible.",
        ],
    },
    "Sleep Disorders": {
        "icon": "🌙",
        "colour": "#7c5cbf",
        "summary": "Your responses suggest disruptions in sleep quality or pattern.",
        "what_it_means": (
            "Sleep disorders encompass a range of difficulties related to getting "
            "adequate, restorative sleep. Your responses indicate you may be "
            "experiencing trouble falling asleep, staying asleep, waking too early, "
            "or consistently poor-quality sleep. Sleep problems both reflect and "
            "worsen other aspects of psychological wellbeing — addressing sleep is "
            "often one of the most effective first steps toward overall improvement."
        ),
        "signs": [
            "Difficulty falling asleep even when tired",
            "Waking frequently during the night or very early in the morning",
            "Feeling unrefreshed after a full night's sleep",
            "Daytime fatigue, difficulty concentrating, or irritability",
            "Relying on screens, food, or substances to fall asleep",
        ],
        "what_to_do": [
            "Keep a consistent sleep and wake time, even at weekends — this is the single most effective habit for improving sleep.",
            "Avoid screens (phone, laptop) for at least 30 minutes before bed.",
            "Keep your bedroom cool, dark, and used only for sleep — not for studying or watching content.",
            "If racing thoughts keep you awake, try writing them down in a notebook before bed to 'offload' them.",
            "If sleep problems persist beyond two to three weeks, speak with a doctor to rule out underlying causes.",
        ],
    },
    "Generalized Anxiety Disorder": {
        "icon": "⚡",
        "colour": "#e6a817",
        "summary": "Your responses reflect elevated anxiety-related behavioural patterns.",
        "what_it_means": (
            "Generalized anxiety involves persistent, difficult-to-control worry that "
            "spans multiple areas of life — not just one specific situation or trigger. "
            "Your responses suggest you may frequently feel on edge, tense, or worried "
            "even when there is no immediate threat, and that this pattern may be "
            "affecting your sleep, concentration, and daily activities. "
            "This is one of the most common and treatable psychological conditions."
        ),
        "signs": [
            "Persistent worry that feels difficult to switch off",
            "Physical tension — tight muscles, headaches, or an unsettled stomach",
            "Difficulty concentrating because your mind keeps going to 'what if' thoughts",
            "Feeling restless, on edge, or easily startled",
            "Avoiding situations that might trigger more worry",
        ],
        "what_to_do": [
            "Practice structured worry time — set aside 15 minutes a day to write down worries, then actively redirect your attention outside that window.",
            "Try diaphragmatic (belly) breathing when anxiety peaks: inhale for 4 counts, hold for 2, exhale for 6.",
            "Challenge catastrophic thinking by asking: 'What is the realistic probability this actually happens?'",
            "Reduce caffeine — it directly amplifies anxiety symptoms.",
            "Cognitive Behavioural Therapy (CBT) is highly effective for generalised anxiety — ask your campus counsellor about it.",
        ],
    },
    "Stress-Related Conditions": {
        "icon": "🔥",
        "colour": "#e05a2b",
        "summary": "Your responses indicate elevated stress levels affecting daily functioning.",
        "what_it_means": (
            "Stress-related conditions arise when the demands placed on you consistently "
            "exceed your current capacity to cope with them. Your responses suggest you "
            "may be carrying a high load of perceived stress — whether from academic "
            "pressure, financial concerns, relationship difficulties, or a combination "
            "of factors. At this level, stress commonly affects sleep, concentration, "
            "physical health, and relationships if not actively managed."
        ),
        "signs": [
            "Feeling overwhelmed by the volume or difficulty of your responsibilities",
            "Irritability, short temper, or snapping at people around you",
            "Difficulty switching off or relaxing even during free time",
            "Physical symptoms such as headaches, jaw tension, or stomach issues",
            "Procrastinating on important tasks because they feel too big to start",
        ],
        "what_to_do": [
            "Write down everything on your plate, then identify the two or three things that actually matter most right now.",
            "Break overwhelming tasks into the smallest possible next step — not 'write essay', but 'open document and write one sentence'.",
            "Build genuine rest into your schedule — not just sleep, but deliberate non-productive time.",
            "Talk to someone about what you're carrying — a friend, family member, or counsellor.",
            "If stress is related to academic workload, speak with your faculty or student support office about available accommodations.",
        ],
    },
    "Eating Disorders": {
        "icon": "🍃",
        "colour": "#2a9d5c",
        "summary": "Your responses suggest patterns related to appetite and eating behaviours.",
        "what_it_means": (
            "Eating-related difficulties can manifest as significant changes in appetite, "
            "a disrupted relationship with food, or eating patterns that feel out of "
            "control. Your responses indicate you may be experiencing notable changes "
            "in how you eat or relate to food, which can both reflect and contribute "
            "to emotional distress. Eating difficulties exist on a wide spectrum and "
            "are more common among university students than is generally recognised."
        ),
        "signs": [
            "Significant unintentional changes in appetite — either loss or increase",
            "Using food as a primary way of coping with stress or difficult emotions",
            "Preoccupation with food, weight, or body image that feels distressing",
            "Eating in ways that feel secretive, guilty, or out of control",
            "Skipping meals regularly due to stress, low mood, or being too busy",
        ],
        "what_to_do": [
            "Try to eat regular meals at consistent times, even if appetite is low — routine helps regulate both eating and mood.",
            "Avoid skipping meals as a stress management strategy — it worsens mood and concentration.",
            "If you notice your relationship with food feeling distressing or out of control, speak with a counsellor or doctor — this is more common than you think and there is effective support available.",
            "The National Alliance for Eating Disorders helpline provides confidential support and resources.",
        ],
    },
    "Cognitive Impairments": {
        "icon": "🧠",
        "colour": "#3a8fc7",
        "summary": "Your responses suggest difficulties with concentration, memory, or mental clarity.",
        "what_it_means": (
            "Cognitive difficulties in this context refer to functional challenges with "
            "attention, concentration, memory, or mental processing — not permanent "
            "neurological damage. Your responses suggest you may be finding it "
            "significantly harder than usual to focus, retain information, or think "
            "clearly. These symptoms are very commonly triggered by sleep deprivation, "
            "high stress, low mood, or anxiety, and tend to improve when those "
            "underlying factors are addressed."
        ),
        "signs": [
            "Difficulty sustaining attention on tasks for more than short periods",
            "Forgetting things more than usual — appointments, instructions, what you just read",
            "Mental fog — a sense that thinking feels slow, effortful, or unclear",
            "Making more mistakes than usual in work or study",
            "Difficulty following conversations or reading longer texts",
        ],
        "what_to_do": [
            "Prioritise sleep above almost everything else — cognitive function degrades rapidly with sleep deprivation.",
            "Work in focused blocks of 25 minutes with 5-minute breaks (the Pomodoro technique) rather than long unbroken sessions.",
            "Reduce multitasking — single-task focus is significantly more effective for people experiencing concentration difficulties.",
            "Exercise, even briefly, has an immediate positive effect on cognitive clarity.",
            "If cognitive difficulties are severe, persistent, or worsening, speak with a doctor to explore underlying causes.",
        ],
    },
    "General Mental Health": {
        "icon": "🌱",
        "colour": "#3aaa6e",
        "summary": "Your responses reflect a broadly stable psychological profile.",
        "what_it_means": (
            "Your responses do not indicate a strong pattern associated with any "
            "specific psychological condition. This is a positive indicator that your "
            "current psychological functioning is broadly stable. General mental health "
            "maintenance — the habits and practices that protect your wellbeing over "
            "time — is just as important as addressing specific difficulties, and this "
            "is a good moment to reinforce those foundations."
        ),
        "signs": [
            "Mood is generally manageable, even if not always positive",
            "Sleep, appetite, and daily functioning are broadly intact",
            "Able to engage with responsibilities and relationships most of the time",
            "Stress is present but generally within a manageable range",
        ],
        "what_to_do": [
            "Keep investing in the habits that got you here — consistent sleep, physical activity, and social connection.",
            "Check in with yourself regularly — a brief weekly reflection on how you're feeling goes a long way.",
            "It's okay to access support before things become difficult — you don't need to be in crisis to speak with a counsellor.",
            "Protect your rest and leisure time deliberately — these aren't rewards for finishing work, they're part of functioning well.",
        ],
    },
    "Coping and Resilience": {
        "icon": "🛡",
        "colour": "#5aab8e",
        "summary": "Your responses suggest active engagement with coping strategies.",
        "what_it_means": (
            "Your responses indicate you are actively using coping strategies to manage "
            "stress and emotional difficulties. Coping and resilience patterns suggest "
            "that while you may be facing challenges, you are engaging with them "
            "constructively rather than avoiding them. Building and maintaining a "
            "diverse set of coping tools is one of the strongest predictors of "
            "long-term psychological wellbeing."
        ),
        "signs": [
            "Actively using strategies like journaling, exercise, or talking to others when stressed",
            "Able to bounce back from setbacks, even if they are difficult initially",
            "Seeking support when needed rather than withdrawing",
            "Maintaining some sense of purpose or direction even during difficult periods",
        ],
        "what_to_do": [
            "Keep diversifying your coping toolkit — relying on only one strategy can leave you vulnerable if that strategy isn't available.",
            "Reflect on what has worked well for you in the past and consciously apply it earlier in difficult periods.",
            "Share your coping strategies with people around you — resilience is often built in community, not isolation.",
            "Continue checking in with yourself regularly so you notice early if your coping strategies stop being sufficient.",
        ],
    },
    "Post-Traumatic Stress Disorder": {
        "icon": "🕊",
        "colour": "#7a7fcf",
        "summary": "Your responses suggest patterns associated with trauma-related stress responses.",
        "what_it_means": (
            "Post-traumatic stress patterns can emerge after experiencing or witnessing "
            "a deeply distressing event. Your responses suggest you may be experiencing "
            "some indicators associated with trauma responses — such as intrusive "
            "memories, heightened alertness, emotional numbing, or avoidance of "
            "reminders of a past event. These are the mind's natural protective "
            "responses to an overwhelming experience, and with the right support, "
            "they are treatable. Please know that reaching out for help is a sign "
            "of strength, not weakness."
        ),
        "signs": [
            "Intrusive memories, flashbacks, or distressing dreams related to a past event",
            "Feeling emotionally numb, detached, or cut off from people and activities",
            "Being easily startled, feeling constantly on guard, or having difficulty relaxing",
            "Avoiding people, places, or situations that remind you of a distressing experience",
            "Intense emotional or physical reactions to reminders of the event",
        ],
        "what_to_do": [
            "Please speak with a mental health professional — trauma-focused therapy (such as EMDR or trauma-focused CBT) is highly effective and you do not need to face this alone.",
            "If you are in immediate distress, contact a crisis line or your campus emergency support service.",
            "Avoid using alcohol or substances to manage symptoms — this provides short-term relief but worsens outcomes over time.",
            "Grounding techniques can help during intrusive moments: focus on 5 things you can see, 4 you can touch, 3 you can hear.",
            "Tell at least one trusted person what you are going through — isolation makes trauma harder to process.",
        ],
    },
}


# ----------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# ----------------------------------------------------------------------
# REGISTER
# ----------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        # --- Basic server-side validation ---
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("register"))

        if User.get_by_username(username) is not None:
            flash("That username is already taken.", "error")
            return redirect(url_for("register"))

        # --- Create the user ---
        User.create(username, email, password, role="user")
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ----------------------------------------------------------------------
# LOGIN
# ----------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.get_by_username(username)

        if user is None or not user.check_password(password):
            flash("Incorrect username or password.", "error")
            return redirect(url_for("login"))

        login_user(user)
        flash(f"Welcome back, {user.username}!", "success")

        if user.role == "admin":
            return redirect(url_for("home"))   # admin dashboard route added later
        return redirect(url_for("home"))        # survey dashboard route added later

    return render_template("login.html")


# ----------------------------------------------------------------------
# LOGOUT
# ----------------------------------------------------------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# ----------------------------------------------------------------------
# SURVEY QUESTION ORDER
# ----------------------------------------------------------------------
# This fixed order controls the order questions appear on the form.
# Short labels are what the user sees; the dict key is the FULL column
# name the predictor.py / model actually expects.
SURVEY_QUESTIONS = [
    {
        "key": "Mood: How would you describe your mood over the past two weeks?",
        "label": "How would you describe your mood over the past two weeks?",
        "type": "nominal",
    },
    {
        "key": "Anxious Social Scale: On a scale of 1-10, how often have you felt anxious in social situations recently?",
        "label": "How often have you felt anxious in social situations recently?",
        "type": "ordinal",
    },
    {
        "key": "Anxiety Triggers: Have you experienced any of the following anxiety triggers in the past month?",
        "label": "Which of these anxiety triggers have you experienced in the past month?",
        "type": "nominal",
    },
    {
        "key": "Sleep Quality: How would you rate the quality of your sleep over the past week?",
        "label": "How would you rate the quality of your sleep over the past week?",
        "type": "nominal",
    },
    {
        "key": "Appetite Change: Have you noticed any significant changes in your appetite?",
        "label": "Have you noticed any significant changes in your appetite?",
        "type": "nominal",
    },
    {
        "key": "Lack of Interest: How often have you felt a lack of interest or pleasure in daily activities?",
        "label": "How often have you felt a lack of interest or pleasure in daily activities?",
        "type": "ordinal",
    },
    {
        "key": "Enjoyable Activities: How often do you engage in activities you enjoy or that help you relax?",
        "label": "How often do you engage in activities you enjoy or that help you relax?",
        "type": "ordinal",
    },
    {
        "key": "Physical Anxiety Symptoms: Have you had any physical symptoms of anxiety (e.g., heart palpitations, sweating, shortness of breath)?",
        "label": "Have you had any physical symptoms of anxiety (e.g. heart palpitations, sweating)?",
        "type": "ordinal",
    },
    {
        "key": "Concentration Difficulty: How often do you find it difficult to concentrate on tasks?",
        "label": "How often do you find it difficult to concentrate on tasks?",
        "type": "ordinal",
    },
    {
        "key": "Coping Strategies: What coping strategies have you used when feeling stressed or anxious?",
        "label": "What coping strategies have you used when feeling stressed or anxious?",
        "type": "nominal",
    },
]


def get_question_options(question):
    """Returns the ordered list of valid answer text for a question dict."""
    if question["type"] == "ordinal":
        mapping = ordinal_maps[question["key"]]
        return sorted(mapping, key=mapping.get)
    else:
        return NOMINAL_OPTIONS[question["key"]]


# ----------------------------------------------------------------------
# SURVEY
# ----------------------------------------------------------------------
@app.route("/survey", methods=["GET", "POST"])
@login_required
def survey():
    if request.method == "POST":
        # Build the answers dict using the SAME full column-name keys
        # that predictor.py expects -- the <select name="..."> values
        # in survey.html are set to these exact keys.
        answers = {}
        for question in SURVEY_QUESTIONS:
            value = request.form.get(question["key"])
            if not value:
                flash("Please answer every question before submitting.", "error")
                return redirect(url_for("survey"))
            answers[question["key"]] = value

        try:
            result = predict_condition(answers)
        except ValueError as e:
            flash(f"Could not process your answers: {e}", "error")
            return redirect(url_for("survey"))

        # --- Save the raw answers ---
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO survey_responses (user_id, answers_json) VALUES (?, ?)",
            (current_user.id, json.dumps(answers))
        )
        conn.commit()
        response_id = cur.lastrowid

        # --- Save the prediction tied to that response ---
        cur = conn.execute(
            "INSERT INTO assessment_results "
            "(response_id, user_id, predicted_condition, confidence_percent, probability_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                response_id,
                current_user.id,
                result["predicted_condition"],
                result["confidence_percent"],
                json.dumps(result["probability_breakdown"]),
            )
        )
        conn.commit()
        result_id = cur.lastrowid
        conn.close()

        return redirect(url_for("result", result_id=result_id))

    questions_with_options = [
        {**q, "options": get_question_options(q)} for q in SURVEY_QUESTIONS
    ]
    return render_template("survey.html", questions=questions_with_options)


# ----------------------------------------------------------------------
# RESULT
# ----------------------------------------------------------------------
@app.route("/result/<int:result_id>")
@login_required
def result(result_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM assessment_results WHERE id = ? AND user_id = ?",
        (result_id, current_user.id)
    ).fetchone()
    conn.close()

    if row is None:
        flash("That result does not exist or does not belong to you.", "error")
        return redirect(url_for("survey"))

    probability_breakdown = json.loads(row["probability_json"])
    condition_info = CONDITION_INFO.get(row["predicted_condition"], {})

    return render_template(
        "result.html",
        predicted_condition=row["predicted_condition"],
        confidence_percent=row["confidence_percent"],
        probability_breakdown=probability_breakdown,
        created_at=row["created_at"],
        condition_info=condition_info,
    )


# ----------------------------------------------------------------------
# HISTORY
# ----------------------------------------------------------------------
@app.route("/history")
@login_required
def history():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, predicted_condition, confidence_percent, created_at "
        "FROM assessment_results WHERE user_id = ? ORDER BY created_at DESC",
        (current_user.id,)
    ).fetchall()
    conn.close()
    return render_template("history.html", results=rows)


# ----------------------------------------------------------------------
# ADMIN DASHBOARD
# ----------------------------------------------------------------------
@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    conn = get_connection()

    total_users = conn.execute(
        "SELECT COUNT(*) AS count FROM users WHERE role = 'user'"
    ).fetchone()["count"]

    total_assessments = conn.execute(
        "SELECT COUNT(*) AS count FROM assessment_results"
    ).fetchone()["count"]

    condition_breakdown = conn.execute(
        "SELECT predicted_condition, COUNT(*) AS count "
        "FROM assessment_results "
        "GROUP BY predicted_condition "
        "ORDER BY count DESC"
    ).fetchall()

    recent_assessments = conn.execute(
        "SELECT u.username, ar.predicted_condition, ar.confidence_percent, ar.created_at "
        "FROM assessment_results ar "
        "JOIN users u ON u.id = ar.user_id "
        "ORDER BY ar.created_at DESC "
        "LIMIT 20"
    ).fetchall()

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_assessments=total_assessments,
        condition_breakdown=condition_breakdown,
        recent_assessments=recent_assessments,
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    init_db()   # safe to call every time -- only creates tables if missing
    app.run(debug=True, port=5000)