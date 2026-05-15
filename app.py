import os
import sqlite3
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
from io import StringIO, BytesIO

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, make_response, jsonify
)

from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ================= CONFIG =================
app.secret_key = os.environ.get('SECRET_KEY', 'change_this_in_production')

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

DATABASE = 'database.db'

# ================= CONSTANTS =================
DEFAULT_MONTHLY_GOAL   = 50_000
RISK_USER_THRESHOLD    = 3
ANOMALY_STD_FACTOR     = 2.0


# ================= INDIAN NUMBER FORMAT =================
def fmt_inr(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "₹0.00"

    negative = value < 0
    value = abs(value)

    int_part = int(value)
    dec_part  = round((value - int_part) * 100)
    dec_str   = f"{dec_part:02d}"

    s = str(int_part)
    if len(s) <= 3:
        formatted = s
    else:
        formatted = s[-3:]
        s = s[:-3]
        while s:
            formatted = s[-2:] + ',' + formatted
            s = s[:-2]

    result = f"{'−' if negative else ''}₹{formatted}.{dec_str}"
    return result


def fmt_inr_plain(value):
    return fmt_inr(value)


@app.template_filter('inr')
def inr_filter(value):
    return fmt_inr(value)


# ================= DB =================
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                allow_admin_view INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                product TEXT NOT NULL,
                category TEXT NOT NULL,
                date TEXT NOT NULL,
                rate REAL NOT NULL,
                cost_price REAL DEFAULT 0,
                quantity INTEGER NOT NULL,
                total REAL NOT NULL,
                profit REAL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        try:
            conn.execute("ALTER TABLE users ADD COLUMN allow_admin_view INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE users ADD COLUMN monthly_goal REAL DEFAULT 50000")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
        except sqlite3.OperationalError:
            pass

        admin = conn.execute('SELECT * FROM users WHERE role="admin"').fetchone()
        if not admin:
            conn.execute("""
                INSERT INTO users (name, email, password, role)
                VALUES (?, ?, ?, ?)
            """, (
                'System Admin',
                'admin@sales.com',
                generate_password_hash('admin123'),
                'admin'
            ))

        conn.commit()


init_db()


# ================= HELPERS =================
def _start_session(user):
    session.permanent = True
    session['user_id'] = user['id']
    session['name']    = user['name']
    session['role']    = user['role']
    session['email']   = user['email']
    session['allow_admin_view'] = bool(user['allow_admin_view'])
    session['monthly_goal'] = user['monthly_goal'] if user['monthly_goal'] else DEFAULT_MONTHLY_GOAL


def generate_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#110407', edgecolor='none', dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def _safe_pct_change(new_val, old_val):
    if old_val == 0:
        return 0.0
    return round((new_val - old_val) / old_val * 100, 1)


def _prep_df(df):
    df = df.copy()
    df['total']   = pd.to_numeric(df['total'],   errors='coerce').fillna(0)
    df['profit']  = pd.to_numeric(df['profit'],  errors='coerce').fillna(0)
    df['date_dt'] = pd.to_datetime(df['date'],   errors='coerce')
    return df


def _get_user_goal(user_id):
    conn = get_db_connection()
    row  = conn.execute("SELECT monthly_goal FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if row and row['monthly_goal']:
        return float(row['monthly_goal'])
    return DEFAULT_MONTHLY_GOAL


# ─────────────────────────────────────────────────────────────────────────────
# 1. AI INSIGHTS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_ai_insights(df):
    if df.empty:
        return ["No sales data found. Add your first sale to unlock insights."]

    insights = []
    today     = pd.Timestamp.now().normalize()
    yesterday = today - pd.Timedelta(days=1)

    today_rev     = df[df['date_dt'] == today]['total'].sum()
    yesterday_rev = df[df['date_dt'] == yesterday]['total'].sum()

    if yesterday_rev > 0:
        day_chg = _safe_pct_change(today_rev, yesterday_rev)
        if day_chg <= -20:
            insights.append(
                f"⚠️ Revenue dropped {abs(day_chg):.1f}% compared to yesterday. "
                "Consider checking for data entry gaps or seasonal dips."
            )
        elif day_chg >= 30:
            insights.append(
                f"🚀 Strong day! Revenue is up {day_chg:.1f}% vs yesterday."
            )

    last_week  = today - pd.Timedelta(days=7)
    prev_week  = today - pd.Timedelta(days=14)
    this_w_rev = df[df['date_dt'] >= last_week]['total'].sum()
    last_w_rev = df[(df['date_dt'] >= prev_week) & (df['date_dt'] < last_week)]['total'].sum()

    if last_w_rev > 0:
        wk_chg = _safe_pct_change(this_w_rev, last_w_rev)
        if wk_chg > 10:
            insights.append(
                f"📈 Weekly revenue grew {wk_chg:.1f}% over the previous week — momentum is building."
            )
        elif wk_chg < -10:
            insights.append(
                f"📉 Weekly revenue fell {abs(wk_chg):.1f}% vs last week. "
                "Review top products for stock or pricing issues."
            )

    total_rev = df['total'].sum()
    total_pft = df['profit'].sum()
    margin    = (total_pft / total_rev * 100) if total_rev > 0 else 0

    if margin < 10:
        insights.append(
            f"🔴 Profit margin is critically low at {margin:.1f}%. "
            "Re-examine cost prices and underperforming SKUs."
        )
    elif margin < 20:
        insights.append(
            f"🟡 Profit margin is {margin:.1f}% — below the healthy 20% benchmark. "
            "Look at your bottom 3 products for margin improvement."
        )
    else:
        insights.append(
            f"✅ Healthy profit margin of {margin:.1f}%. Keep monitoring cost creep."
        )

    cat_rev = df.groupby('category')['total'].sum()
    if not cat_rev.empty:
        best_cat  = cat_rev.idxmax()
        worst_cat = cat_rev.idxmin()
        insights.append(
            f"🏆 '{best_cat}' is your top-earning category "
            f"({fmt_inr(cat_rev[best_cat])}). Double down on it."
        )
        if len(cat_rev) > 1:
            insights.append(
                f"💡 '{worst_cat}' is your weakest category "
                f"({fmt_inr(cat_rev[worst_cat])}). Consider promotions or discontinuation."
            )

    avg_order = total_rev / len(df) if len(df) > 0 else 0
    insights.append(
        f"📦 Average order value is {fmt_inr(avg_order)}. "
        + ("Try bundle deals to push it higher." if avg_order < 500 else "Strong AOV — upselling is working.")
    )

    top_prod = df.groupby('product')['total'].sum().idxmax() if not df.empty else None
    if top_prod:
        insights.append(
            f"🔁 '{top_prod}' is your best-selling product. "
            "Ensure adequate stock and consider a loyalty offer."
        )
    return insights
# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPARISON SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
def generate_comparisons(df):
    if df.empty:
        return {"today_change": 0.0, "week_change": 0.0, "month_change": 0.0}

    today  = pd.Timestamp.now().normalize()

    t_rev  = df[df['date_dt'] == today]['total'].sum()
    y_rev  = df[df['date_dt'] == today - pd.Timedelta(days=1)]['total'].sum()

    tw_rev = df[df['date_dt'] >= today - pd.Timedelta(days=7)]['total'].sum()
    lw_rev = df[
        (df['date_dt'] >= today - pd.Timedelta(days=14)) &
        (df['date_dt'] <  today - pd.Timedelta(days=7))
    ]['total'].sum()

    first_this = today.replace(day=1)
    first_last = (first_this - pd.Timedelta(days=1)).replace(day=1)
    tm_rev = df[df['date_dt'] >= first_this]['total'].sum()
    lm_rev = df[
        (df['date_dt'] >= first_last) &
        (df['date_dt'] <  first_this)
    ]['total'].sum()

    return {
        "today_change": _safe_pct_change(t_rev,  y_rev),
        "week_change":  _safe_pct_change(tw_rev, lw_rev),
        "month_change": _safe_pct_change(tm_rev, lm_rev),
    }
# ─────────────────────────────────────────────────────────────────────────────
# 3. ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies(df):
    alerts = []
    if df.empty or df['date_dt'].isna().all():
        return alerts
    daily = df.groupby('date_dt')['total'].sum().reset_index()
    daily.columns = ['date', 'revenue']

    if len(daily) < 3:
        return alerts

    mean  = daily['revenue'].mean()
    std   = daily['revenue'].std()
    upper = mean + ANOMALY_STD_FACTOR * std
    lower = mean - ANOMALY_STD_FACTOR * std

    for _, row in daily.iterrows():
        date_str = row['date'].strftime('%d %b %Y') if pd.notna(row['date']) else 'Unknown'
        if row['revenue'] > upper:
            alerts.append(
                f"Revenue Spike on {date_str}: {fmt_inr(row['revenue'])} recorded — "
                f"{((row['revenue']-mean)/std):.1f}σ above average ({fmt_inr(mean)}). "
                "Verify entries for duplicates or bulk orders."
            )
        elif row['revenue'] < lower and row['revenue'] >= 0:
            alerts.append(
                f"Revenue Drop on {date_str}: Only {fmt_inr(row['revenue'])} recorded — "
                f"significantly below average ({fmt_inr(mean)}). "
                "Possible missing entries or low-activity day."
            )

    overall_margin = (df['profit'].sum() / df['total'].sum() * 100) if df['total'].sum() > 0 else 0
    if overall_margin < 5:
        alerts.append(
            f"Critical Profit Margin: Overall margin is only {overall_margin:.1f}%. "
            "Immediate review of cost pricing is recommended."
        )

    zero_pft = (df['profit'] <= 0).sum()
    if zero_pft > 0:
        alerts.append(
            f"{zero_pft} Zero/Negative Profit Entries: Some entries are generating no profit. "
            "Review cost prices on those records."
        )
    return alerts
# ─────────────────────────────────────────────────────────────────────────────
# 5. TOP / WORST PRODUCT RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
def get_product_rankings(df):
    if df.empty:
        return [], []

    grouped = df.groupby('product').agg(
        revenue=('total',  'sum'),
        profit= ('profit', 'sum'),
        count=  ('total',  'count'),
    ).reset_index().sort_values('revenue', ascending=False)

    def fmt(row):
        return {
            "name":    row['product'],
            "revenue": fmt_inr(row['revenue']),
            "profit":  fmt_inr(row['profit']),
            "count":   int(row['count']),
        }

    top   = [fmt(r) for _, r in grouped.head(5).iterrows()]
    worst = [fmt(r) for _, r in grouped.tail(3).iloc[::-1].iterrows()]

    return top, worst
# ─────────────────────────────────────────────────────────────────────────────
# 6. TIME-BASED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def get_time_insights(df):
    result = {"best_day": "—", "best_month": "—"}
    if df.empty or df['date_dt'].isna().all():
        return result

    valid = df.dropna(subset=['date_dt']).copy()

    valid['weekday'] = valid['date_dt'].dt.day_name()
    day_rev = valid.groupby('weekday')['total'].sum()
    if not day_rev.empty:
        result['best_day'] = day_rev.idxmax()

    valid['month_label'] = valid['date_dt'].dt.strftime('%b %Y')
    month_rev = valid.groupby('month_label')['total'].sum()
    if not month_rev.empty:
        result['best_month'] = month_rev.idxmax()

    return result
# ─────────────────────────────────────────────────────────────────────────────
# 7. GOAL TRACKING
# ─────────────────────────────────────────────────────────────────────────────
def calculate_goal_progress(df, monthly_goal_target):
    target = float(monthly_goal_target) if monthly_goal_target else DEFAULT_MONTHLY_GOAL

    if df.empty:
        achieved = 0.0
    else:
        today      = pd.Timestamp.now().normalize()
        first_this = today.replace(day=1)
        achieved   = df[df['date_dt'] >= first_this]['total'].sum()

    pct = min(round(achieved / target * 100, 1), 100) if target > 0 else 0

    pft_target   = target * 0.5
    pft_achieved = df [df['date_dt'] >=first_this]['profit'].sum() if not df.empty else 0
    pft_pct      = min(round(pft_achieved / pft_target * 100, 1), 100) if pft_target > 0 else 0

    order_target   = 60
    today_ts       = pd.Timestamp.now().normalize()
    first_m        = today_ts.replace(day=1)
    order_achieved = len(df[df['date_dt'] >= first_m]) if not df.empty else 0
    order_pct      = min(round(order_achieved / order_target * 100, 1), 100)

    goals = [
        {
            "label":   "Monthly Revenue Target",
            "current": fmt_inr(achieved),
            "target":  fmt_inr(target),
            "pct":     pct,
        },
        {
            "label":   "Monthly Profit Goal",
            "current": fmt_inr(pft_achieved),
            "target":  fmt_inr(pft_target),
            "pct":     pft_pct,
        },
        {
            "label":   "Order Volume (this month)",
            "current": str(order_achieved),
            "target":  str(order_target),
            "pct":     order_pct,
        },
    ]

    return {
        "goal_target":           fmt_inr(target),
        "goal_current":          fmt_inr(achieved),
        "goal_achieved_percent": pct,
        "goals":                 goals,
        "goal_raw":              target,
    }
# ─────────────────────────────────────────────────────────────────────────────
# 9. MULTI-CHART DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def prepare_chart_data(df):
    empty = {"trend_chart": [], "profit_vs_revenue": [], "growth_chart": [], "category_chart": []}
    if df.empty:
        return empty

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=29)
    daily  = (
        df[df['date_dt'] >= cutoff]
        .groupby('date_dt')['total']
        .sum()
        .reset_index()
        .sort_values('date_dt')
    )
    trend_chart = [
        {"date": r['date_dt'].strftime('%d %b'), "revenue": round(float(r['total']), 2)}
        for _, r in daily.iterrows()
    ]

    df2 = df.copy()
    df2['month_label'] = df2['date_dt'].dt.strftime('%b %Y')
    monthly = (
        df2.groupby('month_label')
        .agg(revenue=('total', 'sum'), profit=('profit', 'sum'))
        .reset_index()
    )
    profit_vs_revenue = [
        {
            "month":   r['month_label'],
            "revenue": round(float(r['revenue']), 2),
            "profit":  round(float(r['profit']),  2),
        }
        for _, r in monthly.iterrows()
    ]

    if len(monthly) >= 2:
        monthly = monthly.copy()
        monthly['growth'] = monthly['revenue'].pct_change() * 100
        growth_chart = [
            {
                "month":  r['month_label'],
                "growth": round(float(r['growth']), 1) if pd.notna(r['growth']) else 0,
            }
            for _, r in monthly.iterrows()
        ]
    else:
        growth_chart = []

    cat_rev = df.groupby('category')['total'].sum().reset_index()
    cat_rev.columns = ['category', 'revenue']
    cat_rev = cat_rev.sort_values('revenue', ascending=False)
    category_chart = [
        {
            "category": r['category'],
            "revenue":  round(float(r['revenue']), 2),
        }
        for _, r in cat_rev.iterrows()
    ]

    return {
        "trend_chart":       trend_chart,
        "profit_vs_revenue": profit_vs_revenue,
        "growth_chart":      growth_chart,
        "category_chart":    category_chart,
    }
# ─────────────────────────────────────────────────────────────────────────────
# 8. ADMIN INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def _activity_level(entry_count):
    if entry_count >= 10:
        return 'high'
    elif entry_count >= RISK_USER_THRESHOLD:
        return 'medium'
    else:
        return 'low'
def get_admin_intelligence(conn):
    user_rows = conn.execute("""
        SELECT u.id, u.name, u.email,
               COALESCE(SUM(s.total),  0) AS revenue,
               COALESCE(SUM(s.profit), 0) AS profit,
               COUNT(s.id)                AS entry_count
        FROM users u
        LEFT JOIN sales s ON s.user_id = u.id
        WHERE u.role = 'user'
        GROUP BY u.id
        ORDER BY revenue DESC
    """).fetchall()
    leaderboard = []
    top_users   = []
    risk_users  = []

    for rank, row in enumerate(user_rows, start=1):
        entry = {
            "rank":           rank,
            "id":             row['id'],
            "name":           row['name'],
            "email":          row['email'],
            "revenue":        fmt_inr(row['revenue']),
            "profit":         fmt_inr(row['profit']),
            "entry_count":    row['entry_count'],
            "revenue_raw":    float(row['revenue']),
            "activity_level": _activity_level(row['entry_count']),
        }
        leaderboard.append(entry)
        if rank <= 3:
            top_users.append(entry)
        if row['entry_count'] < RISK_USER_THRESHOLD:
            risk_users.append(entry)

    admin_insights = []
    total_rev = sum(r['revenue'] for r in user_rows)
    total_pft = sum(r['profit']  for r in user_rows)

    if total_rev > 0:
        platform_margin = total_pft / total_rev * 100
        admin_insights.append(
            f"Platform profit margin is {platform_margin:.1f}% across all users."
        )
    if risk_users:
        admin_insights.append(
            f"{len(risk_users)} user(s) have fewer than {RISK_USER_THRESHOLD} entries — "
            "they may need onboarding support."
        )
    if leaderboard:
        leader = leaderboard[0]
        admin_insights.append(
            f"Top performer: {leader['name']} with {leader['revenue']} in revenue."
        )
    new_user_count = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='user'"
    ).fetchone()[0]
    admin_insights.append(
        f"Platform has {new_user_count} active user(s) contributing to revenue."
    )
    return {
        "top_users":      top_users,
        "risk_users":     risk_users,
        "leaderboard":    leaderboard,
        "admin_insights": admin_insights,
    }
# ─────────────────────────────────────────────────────────────────────────────
# STATS API
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/stats')
def api_stats():
    conn = get_db_connection()
    active_users = conn.execute(
        'SELECT COUNT(*) FROM users WHERE role != "admin"'
    ).fetchone()[0]
    revenue = conn.execute(
        'SELECT COALESCE(SUM(total),0) FROM sales'
    ).fetchone()[0]
    conn.close()
    return jsonify({"active_users": active_users, "revenue_tracked": revenue})


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form['name']
        email    = request.form['email'].lower().strip()
        password = request.form['password']

        if len(password) < 6:
            flash("Password too short", "danger")
            return render_template('register.html')

        try:
            with get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO users (name, email, password)
                    VALUES (?, ?, ?)
                """, (name, email, generate_password_hash(password)))
                conn.commit()
            flash("Account created!", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "danger")

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email'].lower().strip()
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE email=?', (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            _start_session(user)
            return redirect(
                url_for('admin_dashboard')
                if user['role'] == 'admin'
                else url_for('dashboard')
            )
        flash("Invalid login", "danger")

    return render_template('login.html')
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))
# ─────────────────────────────────────────────────────────────────────────────
# SET GOAL
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/set_goal', methods=['POST'])
def set_goal():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        new_goal = float(request.form.get('monthly_goal', DEFAULT_MONTHLY_GOAL))
        if new_goal <= 0:
            raise ValueError("Goal must be positive")

        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET monthly_goal=? WHERE id=?",
            (new_goal, session['user_id'])
        )
        conn.commit()
        conn.close()

        session['monthly_goal'] = new_goal
        flash(f"Monthly goal updated to {fmt_inr(new_goal)}!", "success")

    except (ValueError, TypeError):
        flash("Invalid goal amount. Please enter a positive number.", "danger")

    return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn    = get_db_connection()
    user_id = session['user_id']

    if request.method == 'POST':
        product  = request.form['product']
        category = request.form['category']
        date     = request.form['date']
        rate     = float(request.form['rate'])
        cost     = float(request.form['cost_price'])
        qty      = int(request.form['qty'])
        total    = rate * qty
        profit   = (rate - cost) * qty

        conn.execute("""
            INSERT INTO sales
            (user_id, product, category, date, rate, cost_price, quantity, total, profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, product, category, date, rate, cost, qty, total, profit))
        conn.commit()
        conn.close()
        flash('Sale added successfully!', 'success')
        return redirect(url_for('dashboard'))

    user_row = conn.execute(
        "SELECT monthly_goal FROM users WHERE id=?", (user_id,)
    ).fetchone()
    monthly_goal_target = float(user_row['monthly_goal']) if user_row and user_row['monthly_goal'] else DEFAULT_MONTHLY_GOAL

    rows = conn.execute(
        'SELECT * FROM sales WHERE user_id=? ORDER BY date DESC',
        (user_id,)
    ).fetchall()

    db_categories = conn.execute(
        "SELECT DISTINCT category FROM sales WHERE user_id=? ORDER BY category",
        (user_id,)
    ).fetchall()
    all_categories = [r['category'] for r in db_categories]

    conn.close()

    df = pd.DataFrame([dict(r) for r in rows])

    stats               = {"total_revenue": "₹0.00", "profit": "₹0.00", "count": 0, "top_product": "N/A", "total_profit": "₹0.00"}
    monthly_profit_data = []
    product_chart       = ""
    category_chart      = ""
    insights            = ["Add sales data to unlock AI insights."]
    comparison          = {"today_change": 0.0, "week_change": 0.0, "month_change": 0.0}
    anomalies           = []
    top_products        = []
    worst_products      = []
    time_insights       = {"best_day": "—", "best_month": "—"}
    goal_data           = {
        "goal_target": fmt_inr(monthly_goal_target),
        "goal_current": "₹0.00",
        "goal_achieved_percent": 0,
        "goals": [],
        "goal_raw": monthly_goal_target,
    }
    chart_data = {"trend_chart": [], "profit_vs_revenue": [], "growth_chart": [], "category_chart": []}

    if not df.empty:
        df = _prep_df(df)

        from_date  = request.args.get('from_date', '').strip()
        to_date    = request.args.get('to_date', '').strip()
        f_category = request.args.get('category', '').strip()
        min_profit = request.args.get('min_profit', '').strip()
        max_profit = request.args.get('max_profit', '').strip()

        if from_date:
            parsed = pd.to_datetime(from_date, errors='coerce')
            if pd.notna(parsed):
                df = df[df['date_dt'] >= parsed]

        if to_date:
            parsed = pd.to_datetime(to_date, errors='coerce')
            if pd.notna(parsed):
                df = df[df['date_dt'] <= parsed]

        if f_category:
            df = df[df['category'].str.strip().str.lower() == f_category.strip().lower()]

        if min_profit:
            try:
                df = df[df['profit'] >= float(min_profit)]
            except ValueError:
                pass

        if max_profit:
            try:
                df = df[df['profit'] <= float(max_profit)]
            except ValueError:
                pass

        if not df.empty:
            stats = {
                "total_revenue":       fmt_inr(df['total'].sum()),
                "profit":              fmt_inr(df['profit'].sum()),
                "total_profit":        fmt_inr(df['profit'].sum()),
                "count":               len(df),
                "top_product":         df.groupby('product')['total'].sum().idxmax(),
                "average_order_value": fmt_inr(df['total'].mean()),
                "profit_margin":       f"{(df['profit'].sum()/df['total'].sum()*100):.1f}%" if df['total'].sum() > 0 else "0.0%",
            }

            df['month'] = df['date_dt'].dt.strftime('%b %Y')
            monthly = (
                df.groupby('month')
                  .agg(revenue=('total', 'sum'), profit=('profit', 'sum'))
                  .reset_index()
            )
            monthly_profit_data = monthly.to_dict('records')

            insights                  = generate_ai_insights(df)
            comparison                = generate_comparisons(df)
            anomalies                 = detect_anomalies(df)
            top_products, worst_products = get_product_rankings(df)
            time_insights             = get_time_insights(df)
            goal_data                 = calculate_goal_progress(df, monthly_goal_target)
            chart_data                = prepare_chart_data(df)

            def style_ax(ax):
                ax.set_facecolor('#0d0305')
                ax.tick_params(colors='#9aa0a6', labelsize=8)
                ax.spines['bottom'].set_color('#1e0608')
                ax.spines['left'].set_color('#1e0608')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            top_p = df.groupby('product')['total'].sum().sort_values(ascending=False).head(6)
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#110407')
            ax.bar(top_p.index, top_p.values, color='#ff2200', alpha=0.85, width=0.5)
            style_ax(ax)
            ax.set_ylabel('Revenue (₹)', color='#9aa0a6', fontsize=8)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            product_chart = generate_chart(fig)
            cat_rev = df.groupby('category')['total'].sum()
            colors  = ['#ff2200', '#cc1a00', '#991300', '#660d00', '#330600', '#ff5533']
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#110407')
            wedges, texts, autotexts = ax.pie(
                cat_rev.values,
                labels=cat_rev.index,
                autopct='%1.1f%%',
                colors=colors[:len(cat_rev)],
                pctdistance=0.8,
                startangle=140
            )
            for t in texts:     t.set_color('#9aa0a6'); t.set_fontsize(8)
            for t in autotexts: t.set_color('#ffffff'); t.set_fontsize(7)
            ax.set_facecolor('#0d0305')
            plt.tight_layout()
            category_chart = generate_chart(fig)

    return render_template(
        'dashboard.html',
        stats               = stats,
        sales               = df.to_dict('records') if not df.empty else [],
        monthly_profit_data = monthly_profit_data,
        product_chart       = product_chart,
        category_chart      = category_chart,
        insights            = insights,
        comparison          = comparison,
        anomalies           = anomalies,
        top_products        = top_products,
        worst_products      = worst_products,
        best_day            = time_insights['best_day'],
        best_month          = time_insights['best_month'],
        goal_target         = goal_data['goal_target'],
        goal_current        = goal_data['goal_current'],
        goal_achieved_percent = goal_data['goal_achieved_percent'],
        goals               = goal_data['goals'],
        goal_raw            = goal_data.get('goal_raw', monthly_goal_target),
        trend_chart         = chart_data['trend_chart'],
        profit_vs_revenue   = chart_data['profit_vs_revenue'],
        growth_chart        = chart_data['growth_chart'],
        category_chart_data = chart_data['category_chart'],
        filter_from         = request.args.get('from_date', ''),
        filter_to           = request.args.get('to_date',   ''),
        filter_category     = request.args.get('category',  ''),
        filter_min_profit   = request.args.get('min_profit', ''),
        filter_max_profit   = request.args.get('max_profit', ''),
        all_categories      = all_categories,
        monthly_goal_target = monthly_goal_target,
    )
# ─────────────────────────────────────────────────────────────────────────────
# EDIT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/edit/<int:id>', methods=['POST'])
def edit_sale(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        product  = request.form['product']
        category = request.form['category']
        date     = request.form['date']
        rate     = float(request.form['rate'])
        cost     = float(request.form['cost_price'])
        qty      = int(request.form['qty'])
        total    = rate * qty
        profit   = (rate - cost) * qty

        conn = get_db_connection()
        conn.execute("""
            UPDATE sales
            SET product=?, category=?, date=?, rate=?, cost_price=?,
                quantity=?, total=?, profit=?
            WHERE id=? AND user_id=?
        """, (product, category, date, rate, cost, qty, total, profit,
              id, session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# DELETE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/delete_sale/<int:id>', methods=['POST'])
def delete_sale(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    conn.execute('DELETE FROM sales WHERE id=? AND user_id=?', (id, session['user_id']))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/export_csv')
def export_csv():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    df   = pd.read_sql_query(
        'SELECT * FROM sales WHERE user_id=?', conn, params=(session['user_id'],)
    )
    conn.close()

    output = StringIO()
    df.to_csv(output, index=False)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=sales.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# PRIVACY TOGGLE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/update_privacy', methods=['POST'])
def update_privacy():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    allow = 'allow_admin' in request.form
    session['allow_admin_view'] = allow

    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET allow_admin_view=? WHERE id=?",
            (1 if allow else 0, session['user_id'])
        )
        conn.commit()
        conn.close()
    except Exception as e:
        flash(f"Could not update privacy setting: {e}", "danger")
        return redirect(url_for('dashboard'))

    flash(
        "Admin access ALLOWED — admins can view your full sales report." if allow
        else "Admin access DENIED — admins can only see your total revenue.",
        "success"
    )
    return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin-panel')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()

    raw_users = conn.execute("""
        SELECT u.id, u.name, u.email, u.role, u.allow_admin_view,
               COALESCE(SUM(s.total), 0)  AS revenue_raw,
               COUNT(s.id)                AS entry_count
        FROM users u
        LEFT JOIN sales s ON s.user_id = u.id
        WHERE u.role = 'user'
        GROUP BY u.id
        ORDER BY u.id
    """).fetchall()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='user'"
    ).fetchone()[0]

    total_sales = conn.execute('SELECT COUNT(*) FROM sales').fetchone()[0]

    totals = conn.execute("""
        SELECT COALESCE(SUM(total),0) AS revenue, COALESCE(SUM(profit),0) AS profit FROM sales
    """).fetchone()

    total_revenue = fmt_inr(totals['revenue'])
    total_profit  = fmt_inr(totals['profit'])

    user_list = []
    for idx, user in enumerate(raw_users, start=1):
        revenue_raw = float(user['revenue_raw'])
        entry_count = user['entry_count']
        user_list.append({
            "id":              idx,
            "db_id":           user["id"],
            "name":            user["name"],
            "email":           user["email"],
            "role":            user["role"],
            "allow_admin_view": user["allow_admin_view"],
            "revenue_raw":     revenue_raw,
            "total_revenue":   fmt_inr(revenue_raw),
            "entry_count":     entry_count,
            "activity_level":  _activity_level(entry_count),
        })

    min_rev  = request.args.get('min_revenue', '').strip()
    max_rev  = request.args.get('max_revenue', '').strip()
    activity = request.args.get('activity', '').strip()
    access   = request.args.get('access', '').strip()

    filtered_users = user_list[:]

    if min_rev:
        try:
            filtered_users = [u for u in filtered_users if u['revenue_raw'] >= float(min_rev)]
        except ValueError:
            pass

    if max_rev:
        try:
            filtered_users = [u for u in filtered_users if u['revenue_raw'] <= float(max_rev)]
        except ValueError:
            pass

    if activity:
        filtered_users = [u for u in filtered_users if u['activity_level'] == activity]

    if access == 'allowed':
        filtered_users = [u for u in filtered_users if u['allow_admin_view']]
    elif access == 'denied':
        filtered_users = [u for u in filtered_users if not u['allow_admin_view']]

    access_allowed = sum(1 for u in user_list if u['allow_admin_view'])
    access_denied  = total_users - access_allowed

    intel = get_admin_intelligence(conn)
    conn.close()

    return render_template(
        'admin-panel.html',
        users          = filtered_users,
        total_users    = total_users,
        total_sales    = total_sales,
        total_revenue  = total_revenue,
        total_profit   = total_profit,
        access_allowed = access_allowed,
        access_denied  = access_denied,
        risk_users     = intel['risk_users'],
        leaderboard    = intel['leaderboard'],
        admin_insights = intel['admin_insights'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — USER DETAIL VIEW
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin/user/<int:user_id>')
def admin_view_user(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()

    if not user or not user['allow_admin_view']:
        conn.close()
        flash("User has restricted access", "danger")
        return redirect(url_for('admin_dashboard'))

    sales = conn.execute(
        'SELECT * FROM sales WHERE user_id=? ORDER BY date DESC', (user_id,)
    ).fetchall()
    conn.close()

    full_data = [dict(row) for row in sales]
    df = pd.DataFrame(full_data) if full_data else pd.DataFrame()
    if not df.empty:
        df = _prep_df(df)

    user_goal = float(user['monthly_goal']) if user['monthly_goal'] else DEFAULT_MONTHLY_GOAL

    if not df.empty:
        total_rev  = df['total'].sum()
        total_pft  = df['profit'].sum()
        avg_order  = df['total'].mean()
        margin_pct = (total_pft / total_rev * 100) if total_rev > 0 else 0
        top_prod   = df.groupby('product')['total'].sum().idxmax()

        stats = {
            "count":               len(df),
            "total_revenue":       fmt_inr(total_rev),
            "total_profit":        fmt_inr(total_pft),
            "profit":              fmt_inr(total_pft),
            "top_product":         top_prod,
            "average_order_value": fmt_inr(avg_order),
            "profit_margin":       f"{margin_pct:.1f}%",
        }

        top_products, worst_products = get_product_rankings(df)
        time_insights  = get_time_insights(df)
        anomalies      = detect_anomalies(df)
        insights       = generate_ai_insights(df)
        chart_data     = prepare_chart_data(df)
        goal_data      = calculate_goal_progress(df, user_goal)
    else:
        stats          = {"count": 0, "total_revenue": "₹0.00", "total_profit": "₹0.00", "profit": "₹0.00", "top_product": "—", "average_order_value": "₹0.00", "profit_margin": "0.0%"}
        top_products   = []
        worst_products = []
        time_insights  = {"best_day": "—", "best_month": "—"}
        anomalies      = []
        insights       = ["No sales data available for this user."]
        chart_data     = {"trend_chart": [], "profit_vs_revenue": [], "growth_chart": [], "category_chart": []}
        goal_data      = {"goals": [], "goal_target": fmt_inr(user_goal), "goal_current": "₹0.00", "goal_achieved_percent": 0}

    return render_template(
        "user-details.html",
        user             = user,
        full_data        = full_data,
        stats            = stats,
        top_products     = top_products,
        worst_products   = worst_products,
        time_analytics   = time_insights,
        anomalies        = anomalies,
        insights         = insights,
        trend_chart      = chart_data['trend_chart'],
        growth_chart     = chart_data['growth_chart'],
        category_chart   = chart_data['category_chart'],
        goals            = goal_data.get('goals', []),
        goal_target      = goal_data.get('goal_target', fmt_inr(user_goal)),
        goal_current     = goal_data.get('goal_current', '₹0.00'),
        goal_achieved_percent = goal_data.get('goal_achieved_percent', 0),
        allow_admin_view = bool(user['allow_admin_view']),
        user_goal_raw    = user_goal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — DELETE USER
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    conn.execute('DELETE FROM sales WHERE user_id=?', (user_id,))
    conn.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE  
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()

    # Fetch fresh user row so email is always current
    user = conn.execute(
        "SELECT id, name, email, role FROM users WHERE id=?",
        (session['user_id'],)
    ).fetchone()

    if not user:
        conn.close()
        session.clear()
        return redirect(url_for('login'))

    stats = conn.execute(
        "SELECT COUNT(*) as count FROM sales WHERE user_id=?",
        (session['user_id'],)
    ).fetchone()

    conn.close()

    return render_template(
        'profile.html',
        user=user,
        stats=stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE PROFILE  
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    name  = request.form.get('name', '').strip()
    email = request.form.get('email', '').lower().strip()

    if not name or not email:
        flash("Name and email are required.", "error")
        return redirect(url_for('profile'))

    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET name=?, email=? WHERE id=?",
            (name, email, session['user_id'])
        )
        conn.commit()

        # Refresh session so navbar and profile page reflect changes immediately
        session['name']  = name
        session['email'] = email

        conn.close()
        flash("Profile updated successfully!", "success")
    except sqlite3.IntegrityError:
        flash("That email address is already in use by another account.", "error")
    except Exception as e:
        flash(f"Update failed: {e}", "error")

    return redirect(url_for('profile'))


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE PASSWORD
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    new_pw  = request.form.get('new_password', '')
    conf_pw = request.form.get('confirm_password', '')

    if len(new_pw) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for('profile'))

    if new_pw != conf_pw:
        flash("Passwords do not match.", "error")
        return redirect(url_for('profile'))

    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_pw), session['user_id'])
    )
    conn.commit()
    conn.close()
    flash("Password updated successfully!", "success")
    return redirect(url_for('profile'))


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('home.html'), 404

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
   app.run(debug=True)
