# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""
import os
import uuid

try:
    from urlparse import urlparse, urljoin
except ImportError:
    from urllib.parse import urlparse, urljoin

import PIL
from PIL import Image
from flask import current_app, request, url_for, redirect, flash
from itsdangerous import BadSignature, SignatureExpired
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

from albumy.extensions import db
from albumy.models import User
from albumy.settings import Operations

import os
import google.generativeai as genai
from ratelimit import limits, sleep_and_retry

from albumy.models import User, Photo, Comment, Collect
from sqlalchemy import func

def generate_token(user, operation, expire_in=None, **kwargs):
    s = Serializer(current_app.config['SECRET_KEY'], expire_in)

    data = {'id': user.id, 'operation': operation}
    data.update(**kwargs)
    return s.dumps(data)


def validate_token(user, token, operation, new_password=None):
    s = Serializer(current_app.config['SECRET_KEY'])

    try:
        data = s.loads(token)
    except (SignatureExpired, BadSignature):
        return False

    if operation != data.get('operation') or user.id != data.get('id'):
        return False

    if operation == Operations.CONFIRM:
        user.confirmed = True
    elif operation == Operations.RESET_PASSWORD:
        user.set_password(new_password)
    elif operation == Operations.CHANGE_EMAIL:
        new_email = data.get('new_email')
        if new_email is None:
            return False
        if User.query.filter_by(email=new_email).first() is not None:
            return False
        user.email = new_email
    else:
        return False

    db.session.commit()
    return True


def rename_image(old_filename):
    ext = os.path.splitext(old_filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


def resize_image(image, filename, base_width):
    filename, ext = os.path.splitext(filename)
    img = Image.open(image)
    if img.size[0] <= base_width:
        return filename + ext
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)

    filename += current_app.config['ALBUMY_PHOTO_SUFFIX'][base_width] + ext
    img.save(os.path.join(current_app.config['ALBUMY_UPLOAD_PATH'], filename), optimize=True, quality=85)
    return filename


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc


def redirect_back(default='main.index', **kwargs):
    for target in request.args.get('next'), request.referrer:
        if not target:
            continue
        if is_safe_url(target):
            return redirect(target)
    return redirect(url_for(default, **kwargs))


def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error in the %s field - %s" % (
                getattr(form, field).label.text,
                error
            ))

def _user_post_metrics(user_id, top_k=3):
    """
    Return top_k posts for user with simple metrics (likes=collects, comments).
    """
    rows = (
        db.session.query(
            Photo.id.label("pid"),
            Photo.description,
            func.count(Collect.collector_id).label("likes"),
            func.count(Comment.id).label("comments"),
        )
        .outerjoin(Collect, Collect.collected_id == Photo.id)
        .outerjoin(Comment, Comment.photo_id == Photo.id)
        .filter(Photo.author_id == user_id)
        .group_by(Photo.id)
        .all()
    )
    # simple score (tune weights in config if you like)
    w = {"collects": 3.0, "comments": 2.5}
    items = []
    for r in rows:
        score = w.get("collects", 3.0) * (r.likes or 0) + w.get("comments", 2.5) * (r.comments or 0)
        items.append({
            "photo_id": r.pid,
            "caption": (r.description or "")[:300],
            "likes": int(r.likes or 0),
            "comments": int(r.comments or 0),
            "score": float(score),
        })
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]

@sleep_and_retry
@limits(calls=5, period=20)
def _gemini_call(img, prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    resp = model.generate_content([img, prompt])
    return (resp.text or "").strip()

def generate_alt_text_gemini(image_path: str, max_len: int = 300) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    img = Image.open(image_path)
    prompt = (
        "Write concise, descriptive HTML alt text (1–2 short sentences). "
        "No camera metadata, no emojis, no 'image of'."
    )
    text = _gemini_call(img, prompt)
    return text[:max_len] if text else ""

@sleep_and_retry
@limits(calls=5, period=60)
def _gemini(parts):
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    resp = model.generate_content(parts)
    return (resp.text or "").strip()

def analyze_user_engagement(user_id: int, top_k: int = 12) -> str:
    """
    One-shot analysis over *all history* for this user. No persistence.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    user = User.query.get_or_404(user_id)
    items = _user_post_metrics(user_id, top_k=top_k)

    if not items:
        return f"No posts for @{user.username} yet — nothing to analyze."

    prompt = (
        "You are a social content analyst. Analyze the user's historical posts.\n"
        "Given post-level metrics (likes, comments) and captions, identify:\n"
        "- recurring themes/topics, "
        "- best-performing content patterns, "
        "- audience takeaways, and "
        "- specific recommendations for future posts.\n"
        "Keep it concise, actionable, and return Markdown with headings and bullets."
    )

    # Build one compact request (cheap + within rate limits)
    parts = [prompt, f"User: @{user.username} ({user.name or ''})"]
    for it in items:
        parts.append(
            f"POST {it['photo_id']}: score={it['score']:.2f}, likes={it['likes']}, "
            f"comments={it['comments']}, caption={it['caption']!r}"
        )

    return _gemini(parts)