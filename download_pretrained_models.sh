#!/bin/bash

# Download pre-trained models from Tactic_Zone project
# These models are trained on football match data

echo "📥 Downloading pre-trained football analysis models..."
echo ""

mkdir -p models

# Model 1: Player detection (players, goalkeepers, goals)
echo "🔍 Downloading player detection model..."
if wget -O models/old_data.pt "https://www.dropbox.com/scl/fi/5wh4yy2ego497sw7ut01y/old_data.pt?rlkey=pkktrpl7kudux5xbaxu2is550&st=ftxxrz0d&dl=1" 2>&1 | tail -3; then
    echo "✅ old_data.pt downloaded"
else
    echo "❌ Failed to download old_data.pt"
fi
echo ""

# Model 2: Event detection (dribbling, tackling, goals, goal line)
echo "⚽ Downloading event detection model..."
if wget -O models/new_data.pt "https://www.dropbox.com/scl/fi/9zf9x3w7r4rizmnn9cbk3/new_data.pt?rlkey=h5gnex1tc0i5egsjpoe1hct5l&st=2araenae&dl=1" 2>&1 | tail -3; then
    echo "✅ new_data.pt downloaded"
else
    echo "❌ Failed to download new_data.pt"
fi
echo ""

# Model 3: Player shirt numbers
echo "👕 Downloading shirt number detection model..."
if wget -O models/playershirt.pt "https://www.dropbox.com/scl/fi/fmkdhhn8aas1jjr3l2xc3/playershirt.pt?rlkey=8kra62hs2fc36p677sm4ms32c&st=70rxlt6a&dl=1" 2>&1 | tail -3; then
    echo "✅ playershirt.pt downloaded"
else
    echo "❌ Failed to download playershirt.pt"
fi
echo ""

# Model 4: Substitution board detection
echo "🔄 Downloading substitution detection model..."
if wget -O models/Substitution.pt "https://www.dropbox.com/scl/fi/638s1flkxaeey0vv2vlrg/Substitution.pt?rlkey=j4720866x3afz53yfzx9xmazo&st=kg4o6z75&dl=1" 2>&1 | tail -3; then
    echo "✅ Substitution.pt downloaded"
else
    echo "❌ Failed to download Substitution.pt"
fi
echo ""

echo "🎉 Download complete!"
echo ""
echo "📁 Models saved in: models/"
echo ""
echo "Model descriptions:"
echo "  • old_data.pt      - Player, goalkeeper, goal detection"
echo "  • new_data.pt      - Events (dribbling, tackling, goals)"
echo "  • playershirt.pt   - Player shirt number detection"
echo "  • Substitution.pt  - Substitution board detection"
