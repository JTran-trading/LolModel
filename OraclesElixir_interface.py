import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import pickle
from sklearn.model_selection import train_test_split

# 1. Load your pickled model (ensure model.pkl is in the same folder)
@st.cache_resource
def load_model():
    with open("OraclesElixir_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

##############Dumping prediction helper functions#######################
# This creates/opens a physical file named 'match_data.db' in your folder
conn = sqlite3.connect("OraclesElixir.db")

#Dumping data

# 1. Load the pre-sanitized modeling dataset directly from SQLite
clean_df = pd.read_sql("SELECT * FROM clean_prematch_dataset", conn)

# 2. Define target and metadata/identifiers to exclude from features
target_col = 'result'
meta_cols = ['gameid', 'result', 'split', 'league']

# 3. Define valid categorical features remaining in the clean dataset
categorical_cols = [c for c in ['league', 'split'] if c in clean_df.columns]

# 4. Gather feature columns dynamically
feature_cols = [c for c in clean_df.columns if c not in meta_cols]

# 5. Convert categorical columns to pandas 'category' dtype for LightGBM
for col in categorical_cols:
    clean_df[col] = clean_df[col].astype('category')

# 6. Handle missing values safely on numeric feature columns
numeric_features = [c for c in feature_cols if clean_df[c].dtype.name != 'category' and pd.api.types.is_numeric_dtype(clean_df[c])]
clean_df[numeric_features] = clean_df[numeric_features].fillna(0)

# 7. Prepare Feature Matrix (X) and Target Vector (y)
X = clean_df[feature_cols]
y = clean_df[target_col]

# 8. Chronological Train/Test Split (no shuffling to prevent future-to-past data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



def get_player_mastery(player_name, champion_name, conn=conn):
    """Pulls individual player-champion synergy stats with safe quote escaping."""
    # Escape single quotes so names like K'sante don't break SQL syntax
    safe_player = str(player_name).replace("'", "''")
    safe_champ = str(champion_name).replace("'", "''")
    
    query = f"""
    SELECT mastery_score, win_rate as champ_win_rate
    FROM player_champions
    WHERE playername = '{safe_player}' AND champion = '{safe_champ}'
    """
    df = pd.read_sql(query, conn)
    if df.empty:
        return {'mastery_score': 0.4, 'champ_win_rate': 0.5}
    return df.iloc[0].to_dict()
def format_match_input_clean(blue_team, red_team, blue_roster_dict, red_roster_dict, league="LCK", model_features=feature_cols, conn=conn):
    """
    Builds a completely clean, zero-artifact inference vector from scratch 
    to guarantee 100% mathematical symmetry.
    """
    # 1. Initialize a dictionary with zeros for all expected model features
    row_data = {col: 0.0 for col in model_features}
    
    # 2. Set categorical metadata if present
    if 'league' in row_data:
        row_data['league'] = league
    if 'split' in row_data:
        row_data['split'] = 'Regular Season'  # Or a default valid category
        
    # 3. Compute Draft Scores and Position Masteries
    blue_draft_score = 0.0
    red_draft_score = 0.0
    
    positions = ['top', 'jng', 'mid', 'bot', 'sup']
    blue_masteries = {}
    red_masteries = {}
    
    for lane in positions:
        # Blue side lookup
        if lane in blue_roster_dict:
            player, champ = blue_roster_dict[lane]
            p_stats = get_player_mastery(player, champ, conn)
            blue_masteries[lane] = float(p_stats['mastery_score'])
            blue_draft_score += float(p_stats['champ_win_rate'])
        else:
            blue_masteries[lane] = 0.0
            
        # Red side lookup
        if lane in red_roster_dict:
            player, champ = red_roster_dict[lane]
            p_stats = get_player_mastery(player, champ, conn)
            red_masteries[lane] = float(p_stats['mastery_score'])
            red_draft_score += float(p_stats['champ_win_rate'])
        else:
            red_masteries[lane] = 0.0

    # 4. Populate Difference Features (Blue minus Red)
    for lane in positions:
        diff_col = f'{lane}_mastery_diff'
        if diff_col in row_data:
            row_data[diff_col] = blue_masteries[lane] - red_masteries[lane]
            
    if 'draft_score_diff' in row_data:
        row_data['draft_score_diff'] = blue_draft_score - red_draft_score

    # 5. Convert to DataFrame matching exact training feature columns
    inference_df = pd.DataFrame([row_data])[model_features]
    
    # 6. Ensure categorical dtypes match training data cleanly
    for col in inference_df.columns:
        if col in X_train.columns and isinstance(X_train[col].dtype, pd.CategoricalDtype):
            train_cats = X_train[col].cat.categories
            val = inference_df[col].iloc[0]
            if val not in train_cats:
                val = train_cats[0]
            inference_df[col] = pd.Categorical([val], categories=train_cats)

    return inference_df


def predict_match_modular(blue_team, red_team, blue_roster_dict, red_roster_dict):
    """
    Evaluates the match from both side perspectives and averages them 
    to guarantee absolute mathematical symmetry.
    """
    # 1. Perspective A: blue_team on Blue, red_team on Red
    df_a = format_match_input_clean(
        blue_team=blue_team, 
        red_team=red_team, 
        blue_roster_dict=blue_roster_dict, 
        red_roster_dict=red_roster_dict
    )
    prob_a_blue = model.predict_proba(df_a)[0][1]  # Prob blue_team wins when starting on Blue
    
    # 2. Perspective B: red_team on Blue, blue_team on Red (Flipped)
    df_b = format_match_input_clean(
        blue_team=red_team, 
        red_team=blue_team, 
        blue_roster_dict=red_roster_dict, 
        red_roster_dict=blue_roster_dict
    )
    prob_b_blue = model.predict_proba(df_b)[0][1]  # Prob red_team wins when starting on Blue
    
    # Convert red_team's blue-side win probability into blue_team's red-side win probability
    prob_a_red_perspective = 1.0 - prob_b_blue
    
    # 3. Average the two perspectives
    final_blue_team_prob = (prob_a_blue + prob_a_red_perspective) / 2.0
    final_red_team_prob = 1.0 - final_blue_team_prob
    
    # 4. Print clean styled output
    print("================ SYMMETRIC MATCH PREDICTION ================")
    print(f"Blue Side: {blue_team}")
    print(f"Red Side:  {red_team}")
    print("-" * 50)
    print(f"Predicted Win Probability:")
    print(f"-> {blue_team}: {final_blue_team_prob * 100:.2f}%")
    print(f"-> {red_team}: {final_red_team_prob * 100:.2f}%")
    print("============================================================")
    
    return final_blue_team_prob, final_red_team_prob


def inspect_matchup_masteries(team_a_name, team_a_roster, team_b_name, team_b_roster, conn=conn):
    """
    Inspects individual player/champion pairs for both teams, calculates mastery 
    and draft scores, and returns the dataframe and totals.
    """
    positions = ['top', 'jng', 'mid', 'bot', 'sup']
    
    team_a_total_draft = 0.0
    team_b_total_draft = 0.0
    
    comparison_data = []
    
    for lane in positions:
        # Team A (Blue) Lookup
        p_a, c_a = team_a_roster.get(lane, ("Unknown", "Unknown"))
        stats_a = get_player_mastery(p_a, c_a, conn)
        team_a_total_draft += stats_a['champ_win_rate']
        
        # Team B (Red) Lookup
        p_b, c_b = team_b_roster.get(lane, ("Unknown", "Unknown"))
        stats_b = get_player_mastery(p_b, c_b, conn)
        team_b_total_draft += stats_b['champ_win_rate']
        
        diff = stats_a['mastery_score'] - stats_b['mastery_score']
        
        comparison_data.append({
            'Lane': lane.upper(),
            f'{team_a_name} (Blue)': f"{p_a} ({c_a})",
            'Mastery (Blue)': f"{stats_a['mastery_score']:.3f}",
            f'{team_b_name} (Red)': f"{p_b} ({c_b})",
            'Mastery (Red)': f"{stats_b['mastery_score']:.3f}",
            'Mastery Diff': f"{diff:+.3f}"
        })

    # Create the DataFrame and return it along with the totals
    comp_df = pd.DataFrame(comparison_data)
    return comp_df, team_a_total_draft, team_b_total_draft


# ==========================================
# STREAMLIT UI CODE
# ==========================================
st.title("League Match Predictor 🎮")
st.write("Configure the teams and rosters below to get a prediction.")

# Team Names Inputs
col1, col2 = st.columns(2)
with col1:
    blue_team = st.text_input("Blue Team Name", value="MVK Esports")
with col2:
    red_team = st.text_input("Red Team Name", value="Karmine Corp")

st.divider()

# Default rosters
default_blue_roster = {
    'top': ('Kratos', "Yone"),
    'jng': ('Gury', "Aatrox"),
    'mid': ('Chika', "Locke"),
    'bot': ('Harky', 'Twitch'),
    'sup': ('SiuLoong', 'Camille')
}

default_red_roster = {
    'top': ('Canna', "Zaheen"),
    'jng': ('Yike', "Nafiri"),
    'mid': ('kyeahoo', "Viktor"),
    'bot': ('Caliste', 'Ezreal'),
    'sup': ('Busio', 'Leona')
}

def build_roster_inputs(team_color, default_roster):
    st.subheader(f"{team_color} Team Roster")
    roster_dict = {}
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    
    for role in roles:
        default_player, default_champ = default_roster[role]
        col_role, col_player, col_champ = st.columns([1, 2, 2])
        
        with col_role:
            st.markdown(f"**{role.upper()}**")
        with col_player:
            player = st.text_input(f"Player ({role})", value=default_player, key=f"{team_color}_{role}_player")
        with col_champ:
            champ = st.text_input(f"Champion ({role})", value=default_champ, key=f"{team_color}_{role}_champ")
            
        roster_dict[role] = (player, champ)
        
    return roster_dict

# Build UI Rosters
blue_roster = build_roster_inputs("Blue", default_blue_roster)
st.markdown("---")
red_roster = build_roster_inputs("Red", default_red_roster)
st.divider()

# Prediction Button and Logic
if st.button("Predict Match Outcome", type="primary", use_container_width=True):
    with st.spinner("Analyzing matchup..."):
        try:
            # 1. Run model prediction
            prediction = predict_match_modular(
                blue_team=blue_team,
                red_team=red_team,
                blue_roster_dict=blue_roster,
                red_roster_dict=red_roster
            )

            # 2. Run mastery inspection and unpack returned variables
            comp_df, team_a_total_draft, team_b_total_draft = inspect_matchup_masteries(
                team_a_name=blue_team, 
                team_a_roster=blue_roster,
                team_b_name=red_team, 
                team_b_roster=red_roster
            )

            # 3. Render results cleanly on mobile & desktop
            st.success("Prediction Complete!")
            
            # Safely unpack model probabilities
            try:
                blue_prob = float(prediction[0])
                red_prob = float(prediction[1])
            except (TypeError, IndexError):
                blue_prob = float(prediction)
                red_prob = 1.0 - blue_prob

            # Model Win Probability as 2-decimal percentages
            st.subheader("🎯 Model Win Probability")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric(label=f"{blue_team} (Blue)", value=f"{blue_prob * 100:.2f}%")
            with col_p2:
                st.metric(label=f"{red_team} (Red)", value=f"{red_prob * 100:.2f}%")

            st.divider()
            st.subheader("📊 Mastery & Draft Breakdown")
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            st.divider()
            st.subheader("📊 Aggregated Draft Score Summary")

            col1, col2 = st.columns(2)
            with col1:
                # Draft score as real number to 4 decimals
                st.metric(label=f"{blue_team} (Blue)", value=f"{(team_a_total_draft / 5):.4f}")
            with col2:
                # Draft score as real number to 4 decimals
                st.metric(label=f"{red_team} (Red)", value=f"{(team_b_total_draft / 5):.4f}")

            # Draft score difference as a real number to 4 decimals
            diff = (team_a_total_draft - team_b_total_draft) / 5
            st.metric(
                label="Draft Score Diff (Blue - Red)",
                value=f"{diff:+.4f}",
                delta=f"{diff:+.4f}",
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")