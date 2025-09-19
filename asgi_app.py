# asgi.py
from asgiref.wsgi import WsgiToAsgi
from backend import app  # imports your Flask app instance named `app`

asgi_app = WsgiToAsgi(app)
