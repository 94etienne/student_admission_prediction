# Student Admission Prediction System 🎓

A machine learning-powered web application that predicts student admission status based on academic performance and other relevant factors. The system supports both REB (Rwanda Education Board) and RTB (Rwanda Technical Board) prediction types.

## 🌟 Features

- **Dual Prediction Types**: Support for both REB and RTB admission predictions
- **Interactive UI**: User-friendly interface with multi-step forms
- **Visual Analytics**: Subject performance charts and score summaries
- **PDF Reports**: Generate downloadable admission reports
- **Real-time Validation**: Instant feedback on eligibility requirements
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Live Demo

Access the live application: [https://studentadmissionprediction.streamlit.app/](https://studentadmissionprediction.streamlit.app/)

## 📋 Prerequisites

Before running the application locally, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🛠️ Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/student_admission_prediction.git
cd student_admission_prediction
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Model Files

```bash
# Create model directory if it doesn't exist
mkdir model

# Ensure you have the following model files in the model/ directory:
# - admission_model.joblib (Trained machine learning model)
# - label_encoders.joblib (Label encoders for categorical variables)
# - program_mapping.joblib (Program mappings for combinations)
# - subject_mapping.joblib (Subject mappings for combinations)
```

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
student_admission_prediction/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── model/                   # Model artifacts directory
│   ├── admission_model.joblib
│   ├── label_encoders.joblib
│   ├── program_mapping.joblib
│   └── subject_mapping.joblib
└── .streamlit/              # Streamlit configuration (optional)
    └── config.toml
```

## 🔧 Dependencies

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
joblib>=1.3.2
pandas>=2.0.3
numpy>=1.24.3
plotly>=5.15.0
reportlab>=4.0.4
scikit-learn>=1.3.0
```

## 💻 Usage

### Usage Guide

#### REB Prediction
```bash
# Steps to follow:
1. Navigate to "REB Prediction" in the sidebar
2. Fill in personal information (NID, name, email, phone)
3. Select your combination and enter scores for 3 principal subjects
4. Provide additional details (completion year, trade skills, program choice)
5. Click "Predict Admission Status"
```

#### RTB Prediction
```bash
# Steps to follow:
1. Navigate to "RTB Prediction" in the sidebar
2. Fill in personal information
3. Select your combination and choose at least 5 subjects with scores
4. Provide additional details including application fee payment status
5. Click "Predict Admission Status"
```

#### Available Features
```bash
# Features you can use:
- Visual Analytics: View subject performance charts
- Score Summary: See individual subject scores and averages
- Program Recommendations: Get recommended programs based on your combination
- PDF Reports: Generate and download detailed admission reports
```

## 🎯 Admission Requirements

### REB Requirements
```
- Before 2024: At least 2 principal passes (≥50%)
- 2024 and after: All subjects must have ≥50%
- Application fee payment (automatically considered paid)
```

### RTB Requirements
```
- Minimum 5 subjects required
- All subjects must have ≥50%
- Application fee must be paid
```

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model files not found**: Check if model files exist in the `model/` directory

3. **Port already in use**: Use a different port
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

4. **Permission errors**: Ensure you have read/write permissions in the project directory

### Error Messages

```bash
# Common error messages and solutions:

"Model files not found"
# Solution: Download or place model files in the correct directory
# mkdir model && cp your_model_files/* model/

"Please fill in all required information"
# Solution: Complete all mandatory fields in the form

"Does not meet minimum academic requirements"
# Solution: Check subject scores and requirements
# REB: At least 2 subjects ≥50% (before 2024) or all subjects ≥50% (2024+)
# RTB: All 5+ subjects ≥50%
```

## 🤝 Contributing

```bash
# How to contribute:
git clone https://github.com/your-username/student_admission_prediction.git
cd student_admission_prediction

# Create a feature branch
git checkout -b feature/AmazingFeature

# Make your changes and commit
git add .
git commit -m 'Add some AmazingFeature'

# Push to the branch
git push origin feature/AmazingFeature

# Open a Pull Request on GitHub
```

## 📊 Model Information

```bash
# The prediction system uses machine learning models trained on:
- Subject combinations and scores
- Completion year and program choice
- Trade skills and additional qualifications
- Application fee payment status
- Institution type (REB vs RTB)

# Model files structure:
model/
├── admission_model.joblib     # Main ML model
├── label_encoders.joblib      # Categorical data encoders
├── program_mapping.joblib     # Combination to program mappings
└── subject_mapping.joblib     # Combination to subject mappings
```

## 🔐 Privacy & Security

```bash
# Security measures implemented:
- Personal information is not stored permanently
- Data is processed locally during the session
- PDF reports are generated client-side
- No data is transmitted to external servers (except for model hosting)
- Session-based data handling only
```

## 📞 Support

```bash
# For technical support or questions:

# GitHub Issues
# Create an issue at: https://github.com/your-username/student_admission_prediction/issues

# Quick troubleshooting commands:
python --version          # Check Python version (should be 3.8+)
pip list                 # Check installed packages
streamlit --version      # Check Streamlit version
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Rwanda Education Board (REB) for academic standards
- Rwanda Technical Board (RTB) for technical education guidelines
- Streamlit team for the excellent web framework
- Contributors and testers who helped improve the system

---

**Note**: This system provides predictions based on historical data and should be used as a guidance tool. Final admission decisions are made by the respective educational institutions.
