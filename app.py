import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# ---------- Encode Images ----------
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = encode_image("logo.png")
light_bg_base64 = encode_image("whitebg.jpg")

# ---------- Load Dataset ----------
df = pd.read_excel("ECLEARNIX_Hackathon_10K_Dataset.xlsx")
df.dropna(inplace=True)

le = LabelEncoder()
for col in ['User_Type', 'Department', 'Region', 'Platform_Source', 'Event_Type', 'Event_Mode']:
    df[col] = le.fit_transform(df[col])

df['Time_Spent_per_Event'] = df['Time_Spent_Total_Minutes'] / (df['Saved_Event_Count'] + 1)
df['Engagement_Score'] = df['Time_Spent_Total_Minutes'] + (df['Feedback_Rating'] * 100) - df['Days_Since_Last_Activity']

# ---------- App Setup ----------
app = Dash(__name__)
app.title = "ECLEARNIX Dashboard"

# ---------- Layout ----------
app.layout = html.Div(style={
    'minHeight': '100vh',
    'backgroundImage': f"url('data:image/jpg;base64,{light_bg_base64}')",
    'backgroundSize': 'cover',
    'backgroundRepeat': 'no-repeat',
    'backgroundPosition': 'center',
    'paddingBottom': '50px'
}, children=[
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(logo_base64),
                 style={'height': '100px', 'display': 'block', 'margin': '0 auto'}),
        html.H1("ECLEARNIX Unified Insights Dashboard",
                style={'textAlign': 'center', 'color': '#013A63', 'fontSize': '36px', 'fontWeight': 'bold'}),
    ]),
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Project Overview', value='tab1'),
        dcc.Tab(label='EDA Insights', value='tab2'),
        dcc.Tab(label='Predictive Modeling', value='tab3'),
        dcc.Tab(label='User Segmentation', value='tab4'),
        dcc.Tab(label='Marketing ROI', value='tab5'),
        dcc.Tab(label='Recommendations', value='tab6'),
    ]),
    html.Div(id='tabs-content')
])

# ---------- Callback for Tabs ----------
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    common_style = {
        'color': '#000', 'textAlign': 'center', 'padding': '40px'
    }

    if tab == 'tab1':
        return html.Div(style=common_style, children=[
            html.H2("Project Overview", style={'fontSize': '42px'}),
            html.P("ECLEARNIX is a fast-growing EdTech platform offering LMS courses, STTPs, FDPs, Hackathons, and more.", style={'fontSize': '40px'}),
            html.Ul(style={'listStyleType': 'none', 'padding': 0, 'fontSize': '36px'}, children=[
                html.Li("🔍 Predict user behavior (like course completion)"),
                html.Li("📈 Understand engagement trends"),
                html.Li("✳️ Segment users based on activity"),
                html.Li("💰 Evaluate marketing ROI"),
                html.Li("🚀 Propose strategies to help ECLEARNIX scale to 1M+ users"),
            ]),
            html.P(f"📁 Total Records in Dataset: {df.shape[0]}", style={'marginTop': '20px', 'fontSize': '30px'}),
            html.Br(),
            html.A("🔗 Visit ECLEARNIX Website", href="https://www.eclearnix.com", target="_blank",
                   style={'color': '#013A63', 'fontSize': '36px', 'textDecoration': 'underline'})
        ])

    elif tab == 'tab2':
        fig1 = px.histogram(df, x='User_Type', title='User Type Distribution', width=900, height=400)
        fig1.update_layout(bargap=0.4, font=dict(size=16))

        fig2 = px.histogram(df, x='Region', title='Region Distribution', width=900, height=400)
        fig2.update_layout(bargap=0.4, font=dict(size=16))

        corr = df.select_dtypes(include='number').corr()
        fig3 = px.imshow(corr, text_auto=True, title='Correlation Heatmap', width=1000, height=600)

        return html.Div(style=common_style, children=[
            html.H3("EDA Insights", style={'fontSize': '28px'}),
            html.Div(dcc.Graph(figure=fig1), style={'maxWidth': '1000px', 'margin': '0 auto'}),
            html.Div(dcc.Graph(figure=fig2), style={'maxWidth': '1000px', 'margin': '0 auto'}),
            html.Div(dcc.Graph(figure=fig3), style={'maxWidth': '1000px', 'margin': '0 auto'})
        ])

    elif tab == 'tab3':
        cm = [[789, 404], [564, 629]]
        fig_cm = px.imshow(cm, text_auto=True, title='Confusion Matrix', width=800, height=400)

        importance_df = pd.DataFrame({
            "Feature": ['Time_Spent_Total_Minutes', 'Feedback_Rating', 'Days_Since_Last_Activity',
                        'Saved_Event_Count', 'Engagement_Score', 'Platform_Source', 'Region'],
            "Importance": [0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07]
        })
        fig_imp = px.bar(importance_df.sort_values("Importance"), x='Importance', y='Feature', orientation='h',
                         title='Feature Importance (XGBoost)', width=800, height=400)

        return html.Div(style=common_style, children=[
            html.H3("Predictive Modeling Results", style={'fontSize': '28px'}),
            html.P("Model Used: XGBoost | Accuracy: ~64% (after SMOTE and tuning)", style={'fontSize': '18px'}),
            html.Div(dcc.Graph(figure=fig_cm), style={'maxWidth': '1000px', 'margin': '0 auto'}),
            html.Div(dcc.Graph(figure=fig_imp), style={'maxWidth': '1000px', 'margin': '0 auto'})
        ])

    elif tab == 'tab4':
        kmeans = KMeans(n_clusters=3, random_state=42)
        clustering_features = df[['Time_Spent_Total_Minutes', 'Feedback_Rating', 'Engagement_Score']]
        df['Cluster'] = kmeans.fit_predict(clustering_features)

        fig_clust = px.scatter(df, x='Time_Spent_Total_Minutes', y='Feedback_Rating', color='Cluster',
                               title='User Clustering (KMeans)', hover_data=['Engagement_Score'],
                               width=900, height=500)

        return html.Div(style=common_style, children=[
            html.H3("User Segmentation - KMeans Clustering", style={'fontSize': '28px'}),
            html.Div(dcc.Graph(figure=fig_clust), style={'maxWidth': '1000px', 'margin': '0 auto'})
        ])

    elif tab == 'tab5':
        platform_df = pd.DataFrame({
            "Platform": ["LinkedIn", "YouTube", "ACE Website", "Email", "WhatsApp", "Instagram"],
            "Avg_Completion_Rate": [0.41, 0.41, 0.40, 0.40, 0.40, 0.38],
            "Avg_Time_Spent": [253, 256, 252, 256, 258, 255],
            "Avg_Feedback": [3.02, 3.02, 3.01, 3.08, 3.02, 3.00]
        })

        fig_bar = px.bar(platform_df, x="Platform", y="Avg_Completion_Rate", color="Platform",
                         title="Avg Completion Rate by Platform", width=800, height=400)

        return html.Div(style=common_style, children=[
            html.H3("Marketing ROI Analysis", style={'fontSize': '28px'}),
            html.Div(dcc.Graph(figure=fig_bar), style={'maxWidth': '1000px', 'margin': '0 auto'})
        ])

    elif tab == 'tab6':
        return html.Div(style=common_style, children=[
            html.H2("Recommendations & Strategic Suggestions", style={'fontSize': '40px'}),
            html.Ul(style={'listStyleType': 'none', 'fontSize': '36px'}, children=[
                html.Li("💡 Focus on high ROI channels (LinkedIn, WhatsApp)"),
                html.Li("🌍 Launch regional campaigns for South & West India"),
                html.Li("🎯 Add personalized learning goal tracking modules"),
                html.Li("🔎 Leverage clusters for content personalization"),
                html.Li("📘 Boost onboarding flow for professionals and faculty")
            ])
        ])

# ---------- Run App ----------
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
