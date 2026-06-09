import logging, os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from engine import engine # Import the memory-efficient engine instance

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Calculate counts using the engine's optimized method
raw_counts = engine.get_geo_counts()
GEO_GROUPS = {
    "african": ["African", "Caribbean"],
    "european": ["UK", "Baltic / Europe", "Scandinavian", "German", "Spanish", "French", "Italian", "Dutch", "Irish", "Scottish", "Portuguese", "Romanian", "Polish", "Austrian", "Belgian"],
    "asian": ["Korean", "Japanese", "Taiwan", "Chinese", "Indonesia", "Malaysian", "Indian", "Arabic", "Russian", "Israeli", "Mongolian"],
    "anglo_americas": ["Canadian", "Australian"],
    "global": ["Global"]
}
CLUSTER_COUNTS = {k: sum(raw_counts.get(r, 0) for r in v) for k, v in GEO_GROUPS.items()}

USER_STATES = {}

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sample_tracks = engine._get_data(columns=['name', 'clean_artists']).sample(n=50).values.tolist()
    user_id = update.effective_chat.id
    USER_STATES[user_id] = {"current_page": 0, "track_pool": sample_tracks, "user_selections": []}
    
    msg_text, reply_markup = generate_page_keyboard(sample_tracks, 0)
    await update.message.reply_text(f"🎵 *Welcome to Bearly Listening!*\n" + msg_text, reply_markup=reply_markup, parse_mode="Markdown")

def generate_page_keyboard(song_list, current_page):
    start_index = current_page * 10
    end_index = min(start_index + 10, len(song_list))
    page_tracks = song_list[start_index:end_index]
    text_line = ["🎵 *Track Catalogue* \n"]
    for index, song in enumerate(page_tracks, start=start_index + 1):
        text_line.append(f"`[{index}]` *{song[0]}* by {song[1]}")
    text_line.append(f"\n📑 *Page {current_page + 1} of {((len(song_list)-1)//10)+1}*")
    
    keyboard = []
    selection_row = []
    for i in range(start_index + 1, end_index + 1):
        selection_row.append(InlineKeyboardButton(text=f"{i}", callback_data=f"rate_{i}"))
        if len(selection_row) == 5:
            keyboard.append(selection_row); selection_row = []
    if selection_row: keyboard.append(selection_row)
    
    keyboard.append([
        InlineKeyboardButton(text="⬅️ Prev", callback_data="nav_prev"),
        InlineKeyboardButton(text="Next ➡️", callback_data="nav_next"),
        InlineKeyboardButton(text="Done ✅", callback_data="nav_done")
    ])
    return "\n".join(text_line), InlineKeyboardMarkup(keyboard)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.message.chat.id
    user_data = USER_STATES.get(user_id)
    if not user_data: return
    
    action = query.data
    track_pool = user_data.get('track_pool', [])
    current_page = user_data.get('current_page', 0)

    if action == 'nav_next':
        max_page = (len(track_pool) - 1) // 10
        if current_page < max_page:
            user_data["current_page"] += 1
            new_text, new_markup = generate_page_keyboard(track_pool, user_data["current_page"])
            await query.edit_message_text(text=new_text, reply_markup=new_markup, parse_mode='Markdown')

    elif action == 'nav_prev':
        if current_page > 0:
            user_data["current_page"] -= 1
            new_text, new_markup = generate_page_keyboard(track_pool, user_data["current_page"])
            await query.edit_message_text(text=new_text, reply_markup=new_markup, parse_mode='Markdown')

    elif action == 'nav_done':
        selections = user_data.get('user_selections', [])
        if not selections:
            await query.answer("⚠️ Please rate at least one track first!", show_alert=True)
            return
        
        user_data['computed_taste_vector'] = engine.calculate_taste_vector(selections)
        await show_dashboard_page_1(query, user_data)
        return

    elif action.startswith("rate_"):
        track_index = int(action.split("_")[1]) - 1
        chosen_track = user_data['track_pool'][track_index]
        user_data['selected_track'] = chosen_track

        rating_keyboard = [
            [
                InlineKeyboardButton(text="1 ⭐", callback_data="score_1"),
                InlineKeyboardButton(text="2 ⭐", callback_data="score_2"),
                InlineKeyboardButton(text="3 ⭐", callback_data="score_3"),
                InlineKeyboardButton(text="4 ⭐", callback_data="score_4"),
                InlineKeyboardButton(text="5 ⭐", callback_data="score_5")
            ],
            [
                InlineKeyboardButton(text="🔙 Back to List", callback_data='score_cancel')
            ]
        ]

        stars_markup = InlineKeyboardMarkup(rating_keyboard)
        rating_prompt_text = f"How many stars do you give *{chosen_track[0]}* by {chosen_track[1]}?"
        await query.edit_message_text(text=rating_prompt_text, reply_markup=stars_markup, parse_mode="Markdown")

    elif action.startswith("score_"):
        if action == 'score_cancel':
            catalog_text, catalog_markup = generate_page_keyboard(track_pool, current_page)
            await query.edit_message_text(text=catalog_text, reply_markup=catalog_markup, parse_mode='Markdown')
            return

        rating_value = int(action.split("_")[1])
        target_song = user_data.get('selected_track')

        if target_song:
            user_data["user_selections"] = [
                x for x in user_data.get("user_selections", [])
                if not (x[0] == target_song[0] and x[1] == target_song[1])
            ]
            user_data["user_selections"].append([target_song[0], target_song[1], rating_value])

            catalog_text, catalog_markup = generate_page_keyboard(track_pool, current_page)
            display_text = f"Choice Saved ✅\n" + catalog_text
            await query.edit_message_text(text=display_text, reply_markup=catalog_markup, parse_mode='Markdown')

    elif action == 'dash_page_1':
        await show_dashboard_page_1(query, user_data)
        return

    elif action == 'select_region':
        n_global = CLUSTER_COUNTS.get("global", 0)
        n_africa = CLUSTER_COUNTS.get("african", 0)
        n_europe = CLUSTER_COUNTS.get("european", 0)
        n_asia = CLUSTER_COUNTS.get("asian", 0)
        n_anglo = CLUSTER_COUNTS.get("anglo_americas", 0)

        region_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(text=f"🌐 Global ({n_global:,} tks)", callback_data="geo_global")],
            [
                InlineKeyboardButton(text=f"🌍 African Wave ({n_africa:,} tks)", callback_data="geo_african"),
                InlineKeyboardButton(text=f"🏰 Euro/UK Focus ({n_europe:,} tks)", callback_data="geo_european")
            ],
            [
                InlineKeyboardButton(text=f"⛩️ Asian Matrix ({n_asia:,} tks)", callback_data="geo_asian"),
                InlineKeyboardButton(text=f"🦘 Anglo-Americas ({n_anglo:,} tks)", callback_data="geo_anglo_americas")
            ],
            [InlineKeyboardButton(text="⬅️ Back to Vibe Matrix", callback_data="dash_page_1")]
        ])
        
        region_text = (
            "🌍 *GEOGRAPHIC CALIBRATION TARGET*\n\n"
            "Where should the system focus your playlist discovery search?\n"
            "The numbers show total tracks available within each data layer cluster."
        )
        await query.edit_message_text(text=region_text, reply_markup=region_keyboard, parse_mode='Markdown')
        return

    elif action.startswith("geo_"):
        chosen_geo = action.split("geo_")[1]
        user_data['target_geo_region'] = chosen_geo
        
        if 'recommended_playlist' in user_data:
            del user_data['recommended_playlist']
            
        user_data['rec_page'] = 0
        await show_dashboard_page_2(query, user_data)
        return

    elif action == 'rec_next':
        recommended_playlist = user_data.get('recommended_playlist', [])
        max_page = (len(recommended_playlist) - 1) // 10
        if user_data.get('rec_page', 0) < max_page:
            user_data['rec_page'] = user_data.get('rec_page', 0) + 1
            await show_dashboard_page_2(query, user_data)
        return

    elif action == 'rec_prev':
        if user_data.get('rec_page', 0) > 0:
            user_data['rec_page'] = user_data.get('rec_page', 0) - 1
            await show_dashboard_page_2(query, user_data)
        return

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text(
            text="🔍 *How to Search:*\nUse `/search song title` or `/search artist name` to filter the music matrix.", 
            parse_mode='Markdown'
        )
        return
    
    search_query = " ".join(context.args)
    status_message = await update.message.reply_text(text="⚡ *Scanning the music matrix...*", parse_mode="Markdown")

    search_result = engine.flexible_two_input_search(search_query)

    if not search_result:
        await status_message.edit_text(text=f"❌ No tracks found matching *'{search_query}'*. Please try a different variant or spelling!", parse_mode='Markdown')
        return
  
    if user_id not in USER_STATES:
        USER_STATES[user_id] = {"user_selections": []}
    
    USER_STATES[user_id]['track_pool'] = search_result
    USER_STATES[user_id]['current_page'] = 0
    USER_STATES[user_id]['current_mode'] = "search"

    msg_text, reply_markup = generate_page_keyboard(search_result, 0)

    await status_message.edit_text(
        text=f"🔍 *Search Matrix Results for '{search_query}':*\n\n" + msg_text,
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def show_dashboard_page_1(query, user_data):
    selections = user_data.get('user_selections', [])
    computed_taste_vector = user_data.get('computed_taste_vector')

    dashboard_lines = [
        "📊 *CALIBRATION MATRIX DASHBOARD* (Page 1/2)\n",
        "*Selected Audio Track Audit:*",
    ]

    rating_descriptions = {
        5: "🔺 2.5x Boost", 4: "🔺 1.5x Boost", 3: "🔹 Neutral", 2: "🔻 -1.5x Suppress", 1: "🔻 -2.5x Suppress"
    }

    for (song_name, artist_name, score) in selections:
        status_label = rating_descriptions.get(score, "🔹 Neutral")
        dashboard_lines.append(f"• *{song_name}* ({score}⭐) => _{status_label}_")

    dashboard_lines.append(f"\n📐 *TARGET AUDIO VECTOR COMPONENTS*")
    dashboard_lines.append("`Feature          | Value   | Visual Spectrum`")
    dashboard_lines.append("`--------------------------------------------`")

    feature_definitions = {
        'danceability': "Suitability for dancing based on tempo, rhythm stability, and beat strength.",
        'energy': "Perceptual measure of intensity, activity, loudness, and chaotic noise.",
        'loudness': "The overall volume of the track in decibels (scaled relative to max intensity).",
        'speechiness': "The presence of spoken words relative to instrumental background layers.",
        'acousticness': "Confidence measure of whether the track uses non-electronic acoustic instruments.",
        'instrumentalness': "Predicts whether a track contains no vocal tracks or lyrics.",
        'liveness': "Detects the presence of an audience or live atmosphere in the recording.",
        'valence': "The musical positiveness conveyed by a track (high valence sounds happy/cheerful).",
        'tempo': "The overall estimated speed or pace of a track measured in Beats Per Minute (BPM)."
    }

    audio_features_list = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    for index, feature_name in enumerate(audio_features_list):
        coordinate_value = computed_taste_vector[index] if computed_taste_vector is not None else 0.0
        filled_block_counts = max(0, min(10, int(coordinate_value * 10)))
        empty_block_count = 10 - filled_block_counts    
        visual_bar_string = ("█" * filled_block_counts) + ("░" * empty_block_count)
        
        data_row_string = f"`{feature_name:<16} | {coordinate_value:.4f}   [{visual_bar_string}]`"
        dashboard_lines.append(data_row_string)

    dashboard_lines.append(f"\n📖 *AUDIO PROFILE GLOSSARY:*")
    for feature_name in audio_features_list:
        desc = feature_definitions[feature_name]
        dashboard_lines.append(f"• *{feature_name.capitalize()}*: _{desc}_")

    dashboard_lines.append(f"\n👉 *Click 'Select Target Region' below to focus your playlist discovery!*")

    navigation_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(text="🌍 Select Target Region ➡️", callback_data="select_region")],
        [InlineKeyboardButton(text="🔄 Restart Fresh Session", callback_data="score_cancel")]
    ])

    await query.edit_message_text(
        text="\n".join(dashboard_lines),
        reply_markup=navigation_keyboard,
        parse_mode='Markdown'
    )

async def show_dashboard_page_2(query, user_data):
    target_geo = user_data.get('target_geo_region', 'global') 
    if 'recommended_playlist' not in user_data:
        # FIX: Call engine directly
        user_data['recommended_playlist'] = engine.generate_geo_recommendations(
            user_data['computed_taste_vector'], region=target_geo
        )

    recommended_playlist = user_data['recommended_playlist']
    
    current_rec_page = user_data.get('rec_page', 0)
    start_index = current_rec_page * 10
    end_index = min(start_index + 10, len(recommended_playlist))
    
    page_tracks = recommended_playlist[start_index:end_index]
    total_pages = ((len(recommended_playlist) - 1) // 10) + 1

    dashboard_lines = [
        f"🔥 *YOUR CUSTOM TARGET RECOMMENDATIONS* ({target_geo.upper()})\n",
        "Here are the top tracks matching your engineered audio vector search profile:\n"
    ]

    for rank_counter, song_row in enumerate(page_tracks, start=start_index + 1):
        rec_song_name, rec_artist_name = song_row[0], song_row[1]
        playlist_line = f"`[{rank_counter:02d}]` 🎵 *{rec_song_name}* \n      └─ _by {rec_artist_name}_"
        dashboard_lines.append(playlist_line)

    dashboard_lines.append(f"\n📑 *Recommendation Page {current_rec_page + 1} of {total_pages}*")
    dashboard_lines.append(f"\n✨ *Calibration Complete! Use /start or click below to generate a new matrix profile.*")

    navigation_keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(text="⬅️ Rec Prev", callback_data="rec_prev"),
            InlineKeyboardButton(text="Rec Next ➡️", callback_data="rec_next")
        ],
        [InlineKeyboardButton(text="🌍 Change Region Focus", callback_data="select_region")],
        [InlineKeyboardButton(text="🔄 Restart Fresh Session", callback_data="score_cancel")]
    ])

    await query.edit_message_text(
        text="\n".join(dashboard_lines),
        reply_markup=navigation_keyboard,
        parse_mode='Markdown'
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = (
        "🐻 **About Bearly-Listening**\n\nBearly-Listening is an experimental backend project focused on building custom recommendation algorithms from scratch. Currently in **Beta v0.1**, this bot serves as a live environment for stress-testing data retrieval and vector-matching logic.\n\n The goal of this project is to move beyond standard API reliance and develop a more nuanced, autonomous recommendation engine."
    )
    await update.message.reply_text(about_text, parse_mode='Markdown')

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info_text = (
        "⚙️ **Technical Specifications & Status**\n\n"
        "• 🎵 **Database:** Currently optimized with **1.08 million local songs**.\n"
        "• 🚧 **API Status:** Beta. Not yet hooked to external APIs (e.g., Spotify/Apple Music).\n"
        "• 🧠 **Recommendation Engine:** Uses **Taste Vector Analysis**.\n"
        "   1. Requests are mapped to a 'Taste Vector' in high-dimensional space.\n"
        "   2. The engine calculates spatial distances to find the best match.\n\n"
        "• 📉 **Limitations:** As a learning project, I may occasionally provide unexpected results. Your feedback is crucial for my training!"
    )
    await update.message.reply_text(info_text, parse_mode='Markdown')

def main() -> None:
    TOKEN = "8881709053:AAHmf9l2cb96Go0TtC3tr8WIrszvGWV9_sE"
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("info", info_command))
    
    # Use Webhook if on Render, else Polling
    if "RENDER_EXTERNAL_URL" in os.environ:
        PORT = int(os.environ.get("PORT", 8000))
        application.run_webhook(
            listen="0.0.0.0", port=PORT, url_path=TOKEN, 
            webhook_url=f"{os.environ['RENDER_EXTERNAL_URL']}/{TOKEN}"
        )
    else:
        application.run_polling()

if __name__ == "__main__":
    main()