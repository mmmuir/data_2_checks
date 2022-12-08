# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv_sp
#     language: python
#     name: python3
# ---

# %%
import json
from glob import glob
from os import path
from random import randint

import pandas as pd
import spotipy
from numpy import nan, where
from ratelimit import limits
from spotipy.oauth2 import SpotifyClientCredentials

# %%
# Instantiate Spotipy
cid = "ec23ca502beb44ffb22173b68cd37d9a"
secret = "556c805ce20848ed94194c081f0c96a8"
sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=cid, client_secret=secret
    )
)


# %%
def get_history():
    json_concat = []
    history = glob(path.join("data", "endsong*.json"))
    for i in range(len(history)):

        if len(history) == 1:

            with open(path.join("data", "endsong.json"), encoding="utf-8") as json_file:
                user_json = json.load(json_file)
                json_concat.append(user_json)
        elif history:
            with open(
                path.join("data", f"endsong_{i}.json"), encoding="utf-8"
            ) as json_file:
                user_json = json.load(json_file)
                json_concat.append(user_json)
        elif not history:
            print(
                "No streaming history in the current working directory. Visit https://www.spotify.com/account/privacy/ to request your extended streaming history and move the endsong.json files to the notebook directory to run analyses on your extended history."
            )
            break
    df = (
        pd.DataFrame([j for i in json_concat for j in i])
        .drop(
            columns=[
                "username",
                "conn_country",
                "ip_addr_decrypted",
                "user_agent_decrypted",
                "platform",
                "incognito_mode",
                "offline_timestamp",
                "offline",
                "skipped",
            ]
        )
        .rename(
            columns={
                "master_metadata_track_name": "track",
                "master_metadata_album_artist_name": "artist",
                "master_metadata_album_album_name": "album",
                "reason_start": "start",
                "reason_end": "end",
                "episode_name": "episode",
                "episode_show_name": "show",
                "spotify_track_uri": "id",
                "duration_ms": "duration",
                "ms_played": "playtime",
                "ts": "timestamp",
            }
        )
        .reset_index(drop=True)
    )
    df["playtime"] = round(df["playtime"].copy() / 1000)
    df["duration"] = round(df["duration"].copy() / 1000)
    df["timestamp"] = pd.to_datetime(df.copy()["timestamp"])
    df["ddate"] = df[["timestamp"]].apply(lambda x: x.dt.date)
    df["dtime"] = df[["timestamp"]].apply(lambda x: x.dt.time)
    df["date"] = df.timestamp.dt.strftime("%m/%d/%Y")
    df["time"] = df.timestamp.dt.strftime("%H:%M:%S")
    df["month"] = df.timestamp.dt.strftime("%b")
    df["year"] = df.timestamp.dt.strftime("%Y")
    df["day"] = df.timestamp.dt.strftime("%a")

    return df



# %%
def get_podcasts(df):
    return (
        df[df["id"].isnull()]
        .reset_index(drop=True)
        .drop(columns=["track", "artist", "album", "id", "shuffle"])
    )



# %%
def remove_podcasts(df):
    # Drop podcast episodes. Reorder columns.
    df = (
        df.fillna(value=nan)  # Todo: keep nan or no
        .loc[df["episode"].isna()]
        .drop(
            columns=[
                "spotify_episode_uri",
                "episode",
                "show",
            ]
        )
    ).reset_index(drop=True)
    return df



# %%
def get_playlist(uri):
    playlist_df = []
    offset = 0
    while True:
        res = sp.playlist_tracks(
            uri,
            offset=offset,
            fields="items.track.id,items.track.artists,items.track.name,items.track.album,total",
        )
        if len(res["items"]) == 0:
            # Combine inner lists and exit loop
            # Todo: ask how this comprehension actually works
            playlist_df = [j for i in playlist_df for j in i]
            print(playlist_df)
            break
        playlist_df.append(res["items"])
        offset = offset + len(res["items"])
        print(offset, "/", res["total"])
    artist_dict = {"artist": [], "track": [], "id": [], "album": []}
    for i in range(len(playlist_df)):
        artist_dict["artist"].append(playlist_df[i]["track"]["artists"][0]["name"])
        artist_dict["track"].append(playlist_df[i]["track"]["name"])
        artist_dict["id"].append(playlist_df[i]["track"]["id"])
        artist_dict["album"].append(playlist_df[i]["track"]["album"]["name"])
    df = pd.DataFrame(artist_dict)
    return df



# %%
def open_wheel():
    with open(path.join("data", "camelot.json")) as json_file:
        camelot_json = json.load(json_file)
        camelot_wheel = pd.DataFrame.from_dict(camelot_json)
        return camelot_wheel



# %%
def key_to_camelot(df):
    df["key"] = (
        df["key"]
        .astype(str)
        .replace(
            {
                "-1": "no key detected",
                "0": "C",
                "1": "D-flat",
                "2": "D",
                "3": "E-flat",
                "4": "E",
                "5": "F",
                "6": "F-sharp",
                "7": "G",
                "8": "A-flat",
                "9": "A",
                "10": "B-flat",
                "11": "B",
            }
        )
    )

    df["mode"] = where(df["mode"] == 1, "major", "minor")
    df["key_signature"] = df["key"] + " " + df["mode"]

    wheel_df = open_wheel().iloc[0]

    # Convert diatonic key signatures to Camelot wheel equivalents.
    df["camelot"] = df["key_signature"].map(
        lambda x: wheel_df.loc[wheel_df == x].index[0]
    )
    df = df.drop(columns=["key", "mode"])



# %%
@limits(calls=200, period=30)
def add_features(df, length=None, playlist=None):
    # Specify length for testing purposes
    df = df[:length]
    # Drop duplicates to limit API calls to include only unique URIs
    df_query = df.drop_duplicates(subset="id")
    offset_min = 0
    offset_max = 50
    af_res_list = []
    while True:
        if offset_min > len(df_query):
            af_res_list = [j for i in af_res_list for j in i]
            merge_cols = (
                pd.DataFrame(af_res_list)
                .loc[:, ["tempo", "duration_ms", "id", "key", "mode"]]
                .rename(
                    columns={
                        "master_metadata_track_name": "track",
                        "master_metadata_album_artist_name": "artist",
                        "master_metadata_album_album_name": "album",
                        "reason_start": "start",
                        "reason_end": "end",
                        "duration_ms": "duration",
                        "ms_played": "playtime",
                        "ts": "timestamp",
                    }
                )
            )
            print(merge_cols)
            key_to_camelot(merge_cols)
            merge_cols = pd.merge(merge_cols, df)
            # Todo: separate function so we can remove these col names from streams_df too in get_history()
            if playlist:
                merge_cols = merge_cols[
                    [
                        "artist",
                        "track",
                        "album",
                        "tempo",
                        "camelot",
                        "key_signature",
                        "id",
                    ]
                ]
            elif not playlist:
                merge_cols = merge_cols[
                    [
                        "artist",
                        "track",
                        "album",
                        "duration",
                        "playtime",
                        "date",
                        "time",
                        "month",
                        "year",
                        "tempo",
                        "camelot",
                        "key_signature",
                        "start",
                        "end",
                        "shuffle",
                        "id",
                        "timestamp",
                    ]
                ]
                merge_cols["date"] = merge_cols["date"].astype(str)
            # Round tempos to nearest whole number for easier. Playlist generation works with tempo ranges, so decimal precision is unnecessary.
            merge_cols["tempo"] = round(merge_cols["tempo"]).astype(
                int
            )  # Todo: delete this if it breaks main
            return merge_cols
        res = sp.audio_features(
            df_query["id"].iloc[offset_min:offset_max],
        )
        if None not in res:
            af_res_list.append(res)
        else:
            res.remove(None)
            af_res_list.append(res)
        offset_min += 50
        offset_max += 50



# %%
# # This version works with uri
# #should also have function to get uri from song title + artist
# #todo: proper type hinting and default values
# # separate functions i suppose, maybe with decorators
# # https://stackoverflow.com/questions/62153371/best-way-to-create-python-function-with-multiple-options


def get_friendly(
    df,
    tempo_range=10,
    uri=None,
    index=None,
    shuffle=None,
    playlist=False,
    shift=["all"],
):
    wheel = open_wheel()
    df = df.drop_duplicates(subset="id").reset_index()
    if uri:
        song_selected = df.loc[df["id"] == uri].iloc[0]
    elif index or index == 0:
        song_selected = df.loc[index]
    elif shuffle:
        song_selected = df.iloc[randint(0, len(df) - 1)]
    else:
        print(
            "Error: no song selected. Specify shuffle=True to operate on random song."
        )
    # Designate desired tempo range
    selected_tempo = song_selected["tempo"]
    acceptable_tempos = range(
        (selected_tempo - tempo_range), (selected_tempo + tempo_range), 1
    )

    # Select harmonically compatible key signatures in camelot.json
    friendly_keys = []
    for i in range(len(shift)):
        key = wheel[song_selected["camelot"]][shift[i]]
        friendly_keys.append(key)
        print(key)
        if type(key) == list:
            friendly_keys.extend(key)

    # Show tracks with harmonically compatible key signatures within a given tempo range. Accounts for Spotify's tendency to double or halve numeric tempos.

    return df.query(
        "camelot in @friendly_keys & (tempo in @acceptable_tempos | tempo * 2 in @acceptable_tempos | tempo / 2 in @acceptable_tempos)"
    )



# %%
def pickl(df, name, all=False):
    return df.to_pickle(path.join("data", name))



# %%
def unpickl(*df):
    for name in df:
        yield pd.read_pickle(path.join("data", name))



# %%
def main():
    # Example playlist
    uri = "spotify:playlist:5CF6KvWn85N6DoWufOjP5T"
    # Todo: delete for production
    testlength = 10

    all_streams_df = get_history()
    podcasts_df = get_podcasts(all_streams_df)
    streams_df = remove_podcasts(all_streams_df)
    streams_af_df = add_features(streams_df, length=testlength)
    playlist_af_df = add_features(get_playlist(uri), length=testlength, playlist=True)
    no_skip_df = streams_af_df.query("(playtime / duration) > 0.75").reset_index(
        drop=True
    )
    wheel_df = open_wheel()

    pickl(streams_df, name="streams_df.p")
    pickl(streams_af_df, name="streams_af_df.p")
    pickl(no_skip_df, name="no_skip_df.p")
    pickl(playlist_af_df, name="playlist_af_df.p")
    pickl(podcasts_df, name="podcasts_df.p")
    pickl(all_streams_df, name="all_streams_df.p")
    pickl(wheel_df, name="wheel_df.p")



# %%
if __name__ == "__main__":
    main()


# %%
# Run this to get runtime statistics and store variables separately from pickle files. %stored variables can be found in
# # %prun -r streams_df, streams_af_df, no_skip_df, playlist_af_df, podcasts_df, all_streams_df, wheel_df = main()
# # %store streams_df streams_af_df no_skip_df playlist_af_df podcasts_df all_streams_df wheel_df
# # %prun -r streams_df, streams_af_df, no_skip_df, playlist_af_df, all_streams_df, wheel_df = main()
# # %store streams_df streams_af_df no_skip_df playlist_af_df all_streams_df wheel_df


# %%
# # %store streams_df streams_af_df no_skip_df playlist_af_df podcasts_df all_streams_df wheel_df

