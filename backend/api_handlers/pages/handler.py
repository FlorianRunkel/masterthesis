# from flask import Blueprint, render_template
# from backend.config import Config

# # Blueprint für die Seiten-Routen erstellen
# pages_bp = Blueprint(
#     'pages_bp', __name__,
#     template_folder=Config.TEMPLATE_DIR,
#     static_folder=Config.STATIC_DIR
# )

# @pages_bp.route('/')
# def index():
#     return render_template('index.html')


# @pages_bp.route('/linkedin')
# def linkedin():
#     return render_template('linkedin.html') 