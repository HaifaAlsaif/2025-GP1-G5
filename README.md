# 🔎 TrustLens  

TrustLens is a web-based platform designed to **detect text authenticity** by distinguishing between **human-written** and **AI-generated** content.  
It empowers researchers, platform owners, and examiners to collaborate on evaluating datasets, creating tasks, and reviewing results with authenticity scoring and explainability insights.  


## 🌐 Overview  

In today’s digital world, AI-generated content is increasingly common, raising challenges in **trust, authenticity, and reliability**.  
TrustLens provides a solution by acting as a **collaborative environment** where:  

- **Project Owners** create and manage projects, upload or generate datasets, and track evaluations.  
- **Examiners** participate in structured tasks (news reviews or human–human / human–AI conversations) and provide feedback.  
- **Admins** initialize the system with seed datasets, AI models, and volunteer examiners.  

This interaction builds a **crowdsourced, evolving dataset**, enabling better AI evaluation and more trustworthy results.  


## ⚙️ Core Functionality  

TrustLens is not just a detector; it’s a **workflow platform** that supports:  

- **Role-based Access** → Dedicated dashboards for Project Owners, Examiners, and Admins.  
- **Project Lifecycle** → Create, edit, delete, and manage projects.  
- **Dataset Management** → Upload structured datasets (CSV) or generate new conversation-based datasets.  
- **Task Management** → Define and assign tasks (News or Conversations).  
- **Evaluation & Explainability** → AI models classify text (Human / AI) with confidence scores and reasoning.  
- **Feedback Integration** → Examiners enrich datasets by submitting feedback and annotations.  
- **System Administration** → Upload seed datasets, register AI models, and onboard volunteers.  


## 🏗️ System Architecture  

TrustLens follows a **client–server architecture** to ensure scalability and modularity:  

- **Frontend (Web App):** Bilingual interface (Arabic / English) for all roles.  
- **Backend (Server):** Manages authentication, datasets, tasks, and project logic.  
- **Database:** Central repository for users, projects, datasets, tasks, and results.  
- **LLM Engine:** Connects to Hugging Face APIs with both baseline and fine-tuned models for classification and authenticity scoring.  

---

## 📌 Project Resources  

- **GitHub Repository:**
  (https://github.com/HaifaAlsaif/2025-GP1-G5.git)  
- **Jira Board:** [[TrustLens-Jira](https://afnanalzakary.atlassian.net/jira/software/projects/WL2025/boards/3/backlog?atlOrigin=eyJpIjoiZTQ0ZGUzMGM0M2Q2NDBlM2I1MDJkZjY2NDI1OGZmZDciLCJwIjoiaiJ9)]  
- **University:** King Saud University – IT496 Graduation Project  
