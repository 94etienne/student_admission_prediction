import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set page config
st.set_page_config(
    page_title="Student Admission Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .success-card {
        background-color: rgba(46, 204, 113, 0.1);
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-card {
        background-color: rgba(231, 76, 60, 0.1);
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-card {
        background-color: #f8f9fa;
        border-left: 5px solid #6a11cb;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_artifacts():
    """Load model and related artifacts"""
    try:
        model = joblib.load('model/admission_model.joblib')
        label_encoders = joblib.load('model/label_encoders.joblib')
        program_map = joblib.load('model/program_mapping.joblib')
        subject_map = joblib.load('model/subject_mapping.joblib')
        return model, label_encoders, program_map, subject_map
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure all model files are in the 'model/' directory")
        return None, None, None, None

def check_admission_eligibility(combination, year, scores, fee_paid, is_tvet=False):
    """Check basic admission eligibility"""
    if not fee_paid:
        return False, []
    
    if is_tvet:
        meets_req = all(score >= 50 for score in scores)
    else:
        if year < 2024:
            principal_passes = sum(score >= 50 for score in scores)
            meets_req = principal_passes >= 2
        else:
            meets_req = all(score >= 50 for score in scores)
    
    if not meets_req:
        return False, []
    
    return True, program_map.get(combination, [])

def prepare_prediction_data(student_data, label_encoders, model):
    """Prepare student data for prediction"""
    student_df_data = {
        'combination': student_data['combination'],
        'completed_year': student_data['completed_year'],
        'has_trade_skills': student_data['has_trade_skills'],
        'application_fee_paid': student_data['application_fee_paid'],
        'program_choice': student_data['program_choice'],
        'is_tvet': student_data.get('is_tvet', 0)
    }
    
    # Add subject scores
    max_subjects = 10
    for i in range(1, max_subjects + 1):
        student_df_data[f'subject{i}'] = 'None'
        student_df_data[f'subject{i}_score'] = 0
    
    for i, (subject, score) in enumerate(student_data['subject_scores']):
        student_df_data[f'subject{i+1}'] = subject
        student_df_data[f'subject{i+1}_score'] = score
    
    student_df = pd.DataFrame([student_df_data])
    
    # Encode categorical variables
    for col in label_encoders:
        if col in student_df.columns:
            try:
                student_df[col] = label_encoders[col].transform(student_df[col])
            except ValueError:
                # Handle unknown categories
                label_encoders[col].classes_ = np.append(
                    label_encoders[col].classes_, 'Unknown'
                )
                student_df[col] = label_encoders[col].transform(student_df[col])
    
    # Ensure all model features are present
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if model_features is not None:
        for feature in model_features:
            if feature not in student_df.columns:
                if '_score' in feature:
                    student_df[feature] = 0
                else:
                    student_df[feature] = 'Unknown'
                    if feature in label_encoders:
                        student_df[feature] = label_encoders[feature].transform(student_df[feature])
        student_df = student_df[model_features]
    
    return student_df

def create_subject_performance_chart(subjects, scores):
    """Create a bar chart for subject performance"""
    colors_list = ['#e74c3c' if score < 50 else '#2ecc71' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=subjects,
            y=scores,
            marker_color=colors_list,
            text=scores,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Subject Performance",
        xaxis_title="Subjects",
        yaxis_title="Scores",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    # Add pass line
    fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                  annotation_text="Pass Mark (50%)")
    
    return fig

def generate_pdf_report(student_info, prediction_result):
    """Generate PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Student Admission Prediction Report", title_style))
    story.append(Spacer(1, 20))
    
    # Student Information
    story.append(Paragraph("Student Information", styles['Heading2']))
    student_data = [
        ['National ID:', student_info.get('nid', 'N/A')],
        ['Name:', f"{student_info.get('fname', '')} {student_info.get('lname', '')}"],
        ['Email:', student_info.get('email', 'N/A')],
        ['Phone:', student_info.get('phone', 'N/A')],
    ]
    
    student_table = Table(student_data)
    student_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(student_table)
    story.append(Spacer(1, 20))
    
    # Prediction Result
    story.append(Paragraph("Admission Prediction", styles['Heading2']))
    status_color = colors.green if prediction_result['admission_status'] == 'Admitted' else colors.red
    story.append(Paragraph(f"Status: {prediction_result['admission_status']}", styles['Normal']))
    story.append(Paragraph(f"Message: {prediction_result['message']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Subject Performance
    story.append(Paragraph("Subject Performance", styles['Heading2']))
    subject_data = [['Subject', 'Score']]
    for subject, score in zip(prediction_result['subject_names'], prediction_result['scores']):
        subject_data.append([subject, f"{score}%"])
    
    subject_table = Table(subject_data)
    subject_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(subject_table)
    story.append(Spacer(1, 20))
    
    # Recommended Programs
    if prediction_result.get('recommended_programs'):
        story.append(Paragraph("Recommended Programs", styles['Heading2']))
        for i, program in enumerate(prediction_result['recommended_programs'], 1):
            story.append(Paragraph(f"{i}. {program}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Load model artifacts
model, label_encoders, program_map, subject_map = load_model_artifacts()

# Check if models loaded successfully
if model is None:
    st.stop()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Admission Prediction System</h1>
    <p>Predict admission status based on academic performance and other factors</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Prediction Type", ["REB Prediction", "RTB Prediction", "About"])

if page == "About":
    st.markdown("""
    <div class="info-card">
        <h2>About This System</h2>
        <p>This Student Admission Prediction system helps students determine their likelihood of admission to various academic programs based on their academic performance and other relevant factors.</p>
        <p>The system uses machine learning algorithms to analyze historical admission data and provide accurate predictions. It considers factors such as:</p>
        <ul>
            <li>Subject scores and combinations</li>
            <li>Program choice and completion year</li>
            <li>Additional qualifications and trade skills</li>
            <li>Application fee payment status</li>
        </ul>
        <p>Results are displayed with detailed analysis and recommendations for alternative programs if needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
elif page in ["REB Prediction", "RTB Prediction"]:
    is_tvet = 1 if page == "RTB Prediction" else 0
    
    # Get combinations
    if is_tvet:
        combinations = ['ACCOUNTING','LSV', 'CET', 'EET', 'MET', 'CP','SoD','AH','MAS',
                      'WOT','FOR','TOR','FOH','MMP','SPE','IND','MPA','NIT','PLT','ETL']
    else:
        combinations = [comb for comb in program_map.keys() if comb not in [
            'ACCOUNTING','LSV', 'CET', 'EET', 'MET', 'CP','SoD','AH','MAS',
            'WOT','FOR','TOR','FOH','MMP','SPE','IND','MPA','NIT','PLT','ETL'
        ]]
    
    # Multi-step form using tabs
    tab1, tab2, tab3 = st.tabs(["📋 Personal Information", "📚 Academic Information", "🎯 Additional Details"])
    
    with tab1:
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            nid = st.text_input("National ID (NID)", placeholder="Enter your National ID")
            fname = st.text_input("First Name", placeholder="Enter your first name")
            email = st.text_input("Email", placeholder="Enter your email address")
        
        with col2:
            lname = st.text_input("Last Name", placeholder="Enter your last name")
            phone = st.text_input("Phone Number", placeholder="Enter your phone number")
    
    with tab2:
        st.subheader("Academic Information")
        
        # Combination selection
        combination = st.selectbox("Select Combination", [""] + combinations)
        
        if combination:
            # Get subjects for selected combination
            subjects_for_combination = subject_map.get(combination, [])
            
            if subjects_for_combination:
                st.write("### Subject Scores")
                
                # Initialize subject scores in session state
                if f'{page.lower()}_subject_scores' not in st.session_state:
                    if is_tvet:
                        st.session_state[f'{page.lower()}_subject_scores'] = {
                            subject: 70 for subject in subjects_for_combination[:5]
                        }
                    else:
                        st.session_state[f'{page.lower()}_subject_scores'] = {
                            subject: 70 for subject in subjects_for_combination[:3]
                        }
                
                subject_scores = {}
                
                if is_tvet:
                    # RTB: Allow adding/removing subjects
                    st.write("Select subjects and enter scores (minimum 5 subjects required):")
                    
                    # Multi-select for subjects
                    selected_subjects = st.multiselect(
                        "Choose Subjects", 
                        subjects_for_combination,
                        default=list(st.session_state[f'{page.lower()}_subject_scores'].keys())
                    )
                    
                    # Score inputs for selected subjects
                    cols = st.columns(2)
                    for i, subject in enumerate(selected_subjects):
                        with cols[i % 2]:
                            score = st.number_input(
                                f"{subject} Score", 
                                min_value=0, 
                                max_value=100, 
                                value=st.session_state[f'{page.lower()}_subject_scores'].get(subject, 70),
                                key=f"rtb_{subject}"
                            )
                            subject_scores[subject] = score
                    
                    if len(selected_subjects) < 5:
                        st.warning("RTB requires at least 5 subjects")
                    
                else:
                    # REB: Fixed 3 principal subjects
                    st.write("Enter scores for your 3 principal subjects:")
                    cols = st.columns(3)
                    
                    for i, subject in enumerate(subjects_for_combination[:3]):
                        with cols[i]:
                            score = st.number_input(
                                f"{subject}", 
                                min_value=0, 
                                max_value=100, 
                                value=st.session_state[f'{page.lower()}_subject_scores'].get(subject, 70),
                                key=f"reb_{subject}"
                            )
                            subject_scores[subject] = score
    
    with tab3:
        st.subheader("Additional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            completed_year = st.number_input("Completion Year", min_value=2010, max_value=2025, value=2024)
            has_trade_skills = st.selectbox("Has Trade Skills", ["No", "Yes"])
        
        with col2:
            if is_tvet:
                application_fee_paid = st.selectbox("Application Fee Paid", ["Yes", "No"])
            else:
                application_fee_paid = "Yes"  # Hidden for REB, default to Yes
            
        # Program selection
        if combination:
            programs = program_map.get(combination, [])
            program_choice = st.selectbox("Program Choice", [""] + programs)
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Admission Status", type="primary", use_container_width=True):
        # Validate inputs
        required_fields = [nid, fname, lname, email, phone, combination]
        if not all(required_fields):
            st.error("Please fill in all required personal information and select a combination.")
        elif not combination:
            st.error("Please select a combination.")
        elif 'subject_scores' not in locals() or not subject_scores:
            st.error("Please enter subject scores.")
        elif is_tvet and len(subject_scores) < 5:
            st.error("RTB requires at least 5 subjects.")
        elif not locals().get('program_choice'):
            st.error("Please select a program choice.")
        else:
            # Prepare data for prediction
            student_data = {
                'combination': combination,
                'completed_year': completed_year,
                'has_trade_skills': 1 if has_trade_skills == "Yes" else 0,
                'application_fee_paid': 1 if application_fee_paid == "Yes" else 0,
                'program_choice': program_choice,
                'is_tvet': is_tvet,
                'subject_scores': list(subject_scores.items())
            }
            
            # Check basic eligibility
            scores = list(subject_scores.values())
            is_eligible, recommended = check_admission_eligibility(
                combination, completed_year, scores, 
                student_data['application_fee_paid'], is_tvet
            )
            
            if not is_eligible:
                st.markdown("""
                <div class="error-card">
                    <h3>❌ Not Admitted</h3>
                    <p>Does not meet minimum academic requirements</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Make prediction
                with st.spinner("Analyzing your application..."):
                    try:
                        student_df = prepare_prediction_data(student_data, label_encoders, model)
                        prediction = model.predict(student_df)[0]
                        
                        # Prepare result
                        result = {
                            'admission_status': 'Admitted' if prediction == 1 else 'Not Admitted',
                            'recommended_programs': recommended,
                            'subject_names': list(subject_scores.keys()),
                            'scores': list(subject_scores.values()),
                            'message': 'Meets academic requirements' if prediction == 1 
                                      else 'Model prediction: Not admitted'
                        }
                        
                        # Display results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if result['admission_status'] == 'Admitted':
                                st.markdown(f"""
                                <div class="success-card">
                                    <h2>🎉 {result['admission_status']}</h2>
                                    <p>{result['message']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="error-card">
                                    <h2>❌ {result['admission_status']}</h2>
                                    <p>{result['message']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Subject performance chart
                            fig = create_subject_performance_chart(
                                result['subject_names'], result['scores']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("📊 Score Summary")
                            for subject, score in subject_scores.items():
                                color = "🟢" if score >= 50 else "🔴"
                                st.write(f"{color} **{subject}**: {score}%")
                            
                            avg_score = np.mean(list(subject_scores.values()))
                            st.metric("Average Score", f"{avg_score:.1f}%")
                        
                        # Recommended programs
                        if result['recommended_programs']:
                            st.subheader("🎯 Recommended Programs")
                            for i, program in enumerate(result['recommended_programs'], 1):
                                st.write(f"{i}. {program}")
                        
                        # PDF generation
                        st.markdown("---")
                        if st.button("📄 Generate PDF Report", type="secondary"):
                            student_info = {
                                'nid': nid, 'fname': fname, 'lname': lname,
                                'email': email, 'phone': phone
                            }
                            
                            pdf_buffer = generate_pdf_report(student_info, result)
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"Admission_Report_{fname}_{lname}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                            
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Student Admission Prediction System - Powered by Etienne NTAMBARA</p>
    <p>For support and inquiries, please contact: +250 783 716 761.</p>
</div>
""", unsafe_allow_html=True)
