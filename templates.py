# templates.py

MODERN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        .section { margin-top: 25px; }
        .skill {
            display: inline-block;
            background: #ecf0f1;
            padding: 6px 12px;
            border-radius: 12px;
            margin: 4px;
        }
    </style>
</head>
<body>

<h1>{{ profile.name }}</h1>
<p>{{ profile.email }} | {{ profile.location }}</p>

<div class="section">
<h3>Professional Summary</h3>
<p>{{ summary }}</p>
</div>

<div class="section">
<h3>Skills</h3>
{% for skill in profile.skills %}
<span class="skill">{{ skill }}</span>
{% endfor %}
</div>

</body>
</html>
"""