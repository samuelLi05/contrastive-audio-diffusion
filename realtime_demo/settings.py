from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "tiny-audio-diffusion-demo-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "realtime_demo.demo",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "realtime_demo.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "realtime_demo" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    }
]

WSGI_APPLICATION = "realtime_demo.wsgi.application"

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "realtime_demo" / "static"]

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "demo_outputs"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TAD_CONFIG_PATH = os.environ.get(
    "TAD_CONFIG_PATH",
    "exp/nsynth_conditional_16gb_embedding_no_wandb.yaml",
)
TAD_CKPT_PATH = os.environ.get(
    "TAD_CKPT_PATH",
    "logs/ckpts/demo-compatible-untrained/epoch=16-valid_loss=0.027.ckpt",
)
TAD_METADATA_PATH = os.environ.get("TAD_METADATA_PATH", "")
TAD_CONDITIONING_MODE = os.environ.get("TAD_CONDITIONING_MODE", "label_embedding")
TAD_DEVICE = os.environ.get("TAD_DEVICE", "")
TAD_CLASS_NAMES = os.environ.get(
    "TAD_CLASS_NAMES",
    "bass,brass,flute,guitar,keyboard,mallet,organ",
)
