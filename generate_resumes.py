import os

resumes = {
    "candidate_david.txt": """Name: David Lee
Contact: david.lee@example.com
Experience:
- 8 years of software development experience.
- Deep expertise in Java, Spring Boot, and microservices architecture.
- Designed and built scalable transaction systems.
- Strong knowledge of SQL databases (PostgreSQL, Oracle).
- Deployed applications on AWS using ECS.

Skills:
- Java
- Spring Boot
- Microservices
- PostgreSQL
- AWS
- JUnit
- Git
""",
    "candidate_eva.txt": """Name: Eva Miller
Contact: eva.miller@example.com
Experience:
- 3 years of frontend development experience.
- Built responsive UI components using React and Next.js.
- Strong understanding of HTML5, CSS3, JavaScript, and Tailwind CSS.
- Experienced with state management libraries like Redux.
- Worked closely with designers to implement pixel-perfect layouts.

Skills:
- JavaScript
- React
- Next.js
- Tailwind CSS
- CSS3/HTML5
- Git
- Figma
""",
    "candidate_frank.txt": """Name: Frank Wilson
Contact: frank.wilson@example.com
Experience:
- 6 years of experience in systems programming.
- Deep knowledge of C++, memory management, and multithreading.
- Built performance-critical desktop applications and low-level libraries.
- Strong command of Linux environment, shell scripting, and Makefile.
- Experience with embedded systems and hardware integration.

Skills:
- C++
- C
- Linux
- Multithreading
- Shell Scripting
- Git
- CMake
""",
    "candidate_grace.txt": """Name: Grace Hopper
Contact: grace.hopper@example.com
Experience:
- 10 years of software engineering experience.
- Expert Python programmer with focus on AI/ML.
- Built and trained deep learning models using PyTorch.
- Developed backend AI services using FastAPI and PostgreSQL.
- Experienced in Docker containerization and Kubernetes orchestration.
- Mentored engineering teams and established coding standards.

Skills:
- Python
- PyTorch
- FastAPI
- PostgreSQL
- Docker
- Kubernetes
- AWS
- AI/ML
""",
    "candidate_henry.txt": """Name: Henry Ford
Contact: henry.ford@example.com
Experience:
- 4 years of QA automation experience.
- Developed test automation frameworks using Python and Selenium.
- Automated API testing with Postman and requests library.
- Managed test suites and integrated them with Jenkins CI/CD pipelines.
- Experienced in writing bug reports and doing manual exploratory testing.

Skills:
- Python
- Selenium
- QA Automation
- API Testing
- Jenkins
- Git
- Postman
""",
    "candidate_ivy.txt": """Name: Ivy Chen
Contact: ivy.chen@example.com
Experience:
- 7 years of product management experience in tech.
- Led cross-functional teams using Agile and Scrum methodologies.
- Defined product vision, roadmap, and requirements for web apps.
- Proficient in JIRA, Confluence, and analytics tools like Mixpanel.
- Collaborated with engineering, design, and marketing to launch products.

Skills:
- Product Management
- Agile & Scrum
- JIRA
- Product Roadmap
- User Research
- Market Analysis
""",
    "candidate_jack.txt": """Name: Jack Ma
Contact: jack.ma@example.com
Experience:
- 5 years of DevOps engineering experience.
- Strong skills in infrastructure as code using Terraform.
- Built and managed Kubernetes clusters on AWS (EKS) and GCP.
- Implemented robust CI/CD pipelines with GitHub Actions.
- Set up monitoring and logging with Prometheus, Grafana, and ELK stack.

Skills:
- AWS
- Kubernetes
- Docker
- Terraform
- CI/CD (GitHub Actions)
- Prometheus
- Grafana
""",
    "candidate_karen.txt": """Name: Karen Page
Contact: karen.page@example.com
Experience:
- 4 years of frontend engineering experience.
- Specialized in React, Next.js, and TypeScript.
- Strong styled-components, CSS, and UI framework experience (Material UI).
- Optimized web applications for maximum speed and SEO.
- Implemented authorization protocols (OAuth, JWT) on frontend.

Skills:
- React
- Next.js
- TypeScript
- Redux
- CSS/Sass
- Git
- OAuth/JWT
""",
    "candidate_leo.txt": """Name: Leo Messi
Contact: leo.messi@example.com
Experience:
- 5 years of experience as a Full Stack Developer.
- Developed backend APIs using Node.js, Express, and NestJS.
- Created interactive interfaces using React and Tailwind CSS.
- Designed database schemas in MongoDB and PostgreSQL.
- Implemented real-time features using WebSockets.

Skills:
- Node.js
- React
- Express
- MongoDB
- PostgreSQL
- Tailwind CSS
- WebSockets
""",
    "candidate_mia.txt": """Name: Mia Hamm
Contact: mia.hamm@example.com
Experience:
- 3 years of experience as a Data Scientist.
- Cleaned and analyzed large datasets using Python, Pandas, and NumPy.
- Wrote complex SQL queries to extract data from warehouse.
- Built predictive models using scikit-learn.
- Designed data visualizations using Tableau and Seaborn.

Skills:
- Python
- Pandas/NumPy
- scikit-learn
- SQL
- Tableau
- Data Visualization
""",
    "candidate_nathan.txt": """Name: Nathan Drake
Contact: nathan.drake@example.com
Experience:
- 4 years of experience as a backend engineer.
- Built enterprise applications using Java, Spring Boot, and Hibernate.
- Experienced with relational databases, specifically PostgreSQL and MySQL.
- Designed RESTful APIs and integrated third-party services.
- Comfortable using Git, Docker, and Maven.

Skills:
- Java
- Spring Boot
- PostgreSQL
- REST APIs
- Docker
- Git
- Maven
""",
    "candidate_olivia.txt": """Name: Olivia Wilde
Contact: olivia.wilde@example.com
Experience:
- 6 years of experience in cybersecurity and application security.
- Performed penetration testing and vulnerability assessments on web apps.
- Solid understanding of OWASP Top 10 vulnerabilities.
- Set up firewall configurations and secure network architectures.
- Assisted developers in fixing security bugs in code.

Skills:
- Cybersecurity
- Penetration Testing
- OWASP Top 10
- Networking
- Linux
- Cryptography
"""
}

import os
os.makedirs("data/resumes", exist_ok=True)
for filename, content in resumes.items():
    filepath = os.path.join("data/resumes", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Generated {len(resumes)} additional mock resumes successfully.")
