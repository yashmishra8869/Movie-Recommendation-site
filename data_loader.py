"""
Generate a compact sample dataset (Hollywood + Bollywood) for the Hybrid Movie Recommendation System.

Usage:
    python data_loader.py
"""

from pathlib import Path
import pandas as pd


def build_sample_movies() -> pd.DataFrame:
    """Create a ready-to-use movie dataset with genres and overviews."""
    movies = [
        {"title": "Inception", "genres": "Sci-Fi|Thriller|Action", "overview": "A thief enters dreams to plant ideas while facing his own past.", "tmdb_id": 27205},
        {"title": "Interstellar", "genres": "Sci-Fi|Drama|Adventure", "overview": "Explorers travel through a wormhole to save humanity from extinction.", "tmdb_id": 157336},
        {"title": "The Dark Knight", "genres": "Action|Crime|Drama", "overview": "Batman faces the Joker as chaos engulfs Gotham City.", "tmdb_id": 155},
        {"title": "The Matrix", "genres": "Sci-Fi|Action", "overview": "A hacker discovers reality is a simulation and joins a rebellion.", "tmdb_id": 603},
        {"title": "The Shawshank Redemption", "genres": "Drama|Crime", "overview": "A banker forms an enduring friendship while serving a life sentence.", "tmdb_id": 278},
        {"title": "Fight Club", "genres": "Drama|Thriller", "overview": "An office worker starts an underground club that grows beyond control.", "tmdb_id": 550},
        {"title": "Forrest Gump", "genres": "Drama|Romance", "overview": "A kind-hearted man witnesses key moments in modern American history.", "tmdb_id": 13},
        {"title": "Pulp Fiction", "genres": "Crime|Drama", "overview": "Intertwined crime stories unfold in nonlinear fashion in Los Angeles.", "tmdb_id": 680},
        {"title": "The Godfather", "genres": "Crime|Drama", "overview": "A mafia family transitions power to the next generation.", "tmdb_id": 238},
        {"title": "The Godfather Part II", "genres": "Crime|Drama", "overview": "Parallel stories show a young Don and his son maintaining empire.", "tmdb_id": 240},
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "genres": "Fantasy|Adventure|Action", "overview": "A hobbit begins a quest to destroy a ring of immense power.", "tmdb_id": 120},
        {"title": "The Lord of the Rings: The Two Towers", "genres": "Fantasy|Adventure|Action", "overview": "The fellowship is broken while war rises across Middle-earth.", "tmdb_id": 121},
        {"title": "The Lord of the Rings: The Return of the King", "genres": "Fantasy|Adventure|Action", "overview": "The final battle for Middle-earth decides the fate of all.", "tmdb_id": 122},
        {"title": "Avatar", "genres": "Sci-Fi|Adventure|Action", "overview": "A marine on Pandora is torn between duty and a new world.", "tmdb_id": 19995},
        {"title": "Titanic", "genres": "Romance|Drama", "overview": "A love story unfolds aboard the doomed ocean liner Titanic.", "tmdb_id": 597},
        {"title": "Gladiator", "genres": "Action|Drama|History", "overview": "A Roman general seeks justice after betrayal and tragedy.", "tmdb_id": 98},
        {"title": "The Avengers", "genres": "Action|Sci-Fi|Adventure", "overview": "Earth's mightiest heroes unite against an alien invasion.", "tmdb_id": 24428},
        {"title": "Avengers: Infinity War", "genres": "Action|Sci-Fi|Adventure", "overview": "The Avengers battle Thanos before he collects all infinity stones.", "tmdb_id": 299536},
        {"title": "Avengers: Endgame", "genres": "Action|Sci-Fi|Adventure", "overview": "Surviving heroes attempt a final mission to undo devastation.", "tmdb_id": 299534},
        {"title": "Iron Man", "genres": "Action|Sci-Fi", "overview": "A billionaire engineer builds a powered suit after captivity.", "tmdb_id": 1726},
        {"title": "Doctor Strange", "genres": "Action|Fantasy|Sci-Fi", "overview": "A surgeon discovers mystic arts after a career-ending accident.", "tmdb_id": 284052},
        {"title": "Black Panther", "genres": "Action|Sci-Fi|Adventure", "overview": "The king of Wakanda faces a rival with a claim to the throne.", "tmdb_id": 284054},
        {"title": "Spider-Man: Homecoming", "genres": "Action|Sci-Fi|Adventure", "overview": "A young Spider-Man balances school life and superhero duties.", "tmdb_id": 315635},
        {"title": "Spider-Man: No Way Home", "genres": "Action|Sci-Fi|Adventure", "overview": "Spider-Man's identity exposure causes multiverse chaos.", "tmdb_id": 634649},
        {"title": "Joker", "genres": "Crime|Drama|Thriller", "overview": "A struggling comedian descends into madness in Gotham.", "tmdb_id": 475557},
        {"title": "Parasite", "genres": "Thriller|Drama", "overview": "A poor family infiltrates a wealthy household with unexpected consequences.", "tmdb_id": 496243},
        {"title": "Whiplash", "genres": "Drama|Music", "overview": "A drummer endures extreme training under a relentless instructor.", "tmdb_id": 244786},
        {"title": "La La Land", "genres": "Romance|Drama|Music", "overview": "A musician and actress fall in love while chasing dreams.", "tmdb_id": 313369},
        {"title": "Mad Max: Fury Road", "genres": "Action|Adventure|Sci-Fi", "overview": "Survivors flee a tyrant across a post-apocalyptic wasteland.", "tmdb_id": 76341},
        {"title": "Dune", "genres": "Sci-Fi|Adventure|Drama", "overview": "A gifted heir navigates politics and destiny on a desert planet.", "tmdb_id": 438631},
        {"title": "Dune: Part Two", "genres": "Sci-Fi|Adventure|Drama", "overview": "Paul unites desert tribes to challenge galactic powers.", "tmdb_id": 693134},
        {"title": "The Social Network", "genres": "Drama", "overview": "The rise of Facebook sparks legal and personal conflicts.", "tmdb_id": 37799},
        {"title": "The Imitation Game", "genres": "Drama|History|Thriller", "overview": "Alan Turing leads codebreakers during World War II.", "tmdb_id": 205596},
        {"title": "The Prestige", "genres": "Drama|Mystery|Thriller", "overview": "Two rival magicians obsess over outperforming each other.", "tmdb_id": 1124},
        {"title": "Memento", "genres": "Mystery|Thriller", "overview": "A man with memory loss hunts his wife's killer.", "tmdb_id": 77},
        {"title": "Shutter Island", "genres": "Thriller|Mystery", "overview": "A marshal investigates a psychiatric hospital disappearance.", "tmdb_id": 11324},
        {"title": "The Silence of the Lambs", "genres": "Thriller|Crime", "overview": "An FBI trainee seeks help from a brilliant imprisoned killer.", "tmdb_id": 274},
        {"title": "Se7en", "genres": "Crime|Thriller|Mystery", "overview": "Detectives hunt a serial killer inspired by seven deadly sins.", "tmdb_id": 807},
        {"title": "The Departed", "genres": "Crime|Thriller|Drama", "overview": "An undercover cop and a mole race to expose each other.", "tmdb_id": 1422},
        {"title": "Good Will Hunting", "genres": "Drama|Romance", "overview": "A gifted janitor confronts his past through therapy.", "tmdb_id": 489},
        {"title": "The Grand Budapest Hotel", "genres": "Comedy|Drama|Adventure", "overview": "A concierge and lobby boy become entangled in a murder mystery.", "tmdb_id": 120467},
        {"title": "The Wolf of Wall Street", "genres": "Comedy|Crime|Drama", "overview": "A stockbroker's rise is fueled by greed and excess.", "tmdb_id": 106646},
        {"title": "The Truman Show", "genres": "Drama|Comedy", "overview": "A man discovers his life is a globally broadcast reality show.", "tmdb_id": 37165},
        {"title": "The Revenant", "genres": "Drama|Adventure|Western", "overview": "A frontiersman fights for survival after a brutal betrayal.", "tmdb_id": 281957},
        {"title": "1917", "genres": "War|Drama", "overview": "Two soldiers race against time to deliver a critical message.", "tmdb_id": 530915},
        {"title": "Top Gun: Maverick", "genres": "Action|Drama", "overview": "Maverick trains elite pilots for a high-risk mission.", "tmdb_id": 361743},
        {"title": "Oppenheimer", "genres": "Drama|History", "overview": "The story of J. Robert Oppenheimer and the atomic bomb project.", "tmdb_id": 872585},
        {"title": "Barbie", "genres": "Comedy|Adventure|Fantasy", "overview": "Barbie leaves Barbieland for a journey of identity.", "tmdb_id": 346698},
        {"title": "Everything Everywhere All at Once", "genres": "Sci-Fi|Comedy|Action", "overview": "A woman explores multiple universes to save reality and family.", "tmdb_id": 545611},
        {"title": "The Batman", "genres": "Crime|Mystery|Action", "overview": "Batman uncovers corruption while tracking a serial killer.", "tmdb_id": 414906},

        # Bollywood additions
        {"title": "3 Idiots", "genres": "Comedy|Drama", "overview": "Three engineering students navigate friendship, pressure, and the meaning of success.", "tmdb_id": None},
        {"title": "Dangal", "genres": "Drama|Sport|Biography", "overview": "A former wrestler trains his daughters to become world-class champions.", "tmdb_id": None},
        {"title": "PK", "genres": "Comedy|Drama|Sci-Fi", "overview": "An alien questions social beliefs and human behavior in India.", "tmdb_id": None},
        {"title": "Bajrangi Bhaijaan", "genres": "Drama|Adventure", "overview": "A devoted man helps a lost child reunite with her family across borders.", "tmdb_id": None},
        {"title": "Lagaan", "genres": "Drama|Sport|History", "overview": "Villagers challenge British officers to a cricket match to avoid heavy taxes.", "tmdb_id": None},
        {"title": "Swades", "genres": "Drama", "overview": "A scientist returns to India and rediscovers purpose through rural change.", "tmdb_id": None},
        {"title": "Taare Zameen Par", "genres": "Drama|Family", "overview": "A teacher helps a child with dyslexia unlock his true potential.", "tmdb_id": None},
        {"title": "Zindagi Na Milegi Dobara", "genres": "Drama|Comedy|Adventure", "overview": "Three friends take a life-changing road trip across Spain.", "tmdb_id": None},
        {"title": "Queen", "genres": "Comedy|Drama", "overview": "A woman embarks on a solo honeymoon and discovers independence.", "tmdb_id": None},
        {"title": "Andhadhun", "genres": "Crime|Thriller|Comedy", "overview": "A pianist pretending to be blind becomes entangled in a murder mystery.", "tmdb_id": None},
        {"title": "Gangs of Wasseypur", "genres": "Crime|Drama|Action", "overview": "A multi-generational crime saga unfolds through rivalry and revenge.", "tmdb_id": None},
        {"title": "Kabir Singh", "genres": "Romance|Drama", "overview": "A brilliant but troubled surgeon spirals after heartbreak.", "tmdb_id": None},
        {"title": "War", "genres": "Action|Thriller", "overview": "An elite soldier is assigned to hunt his former mentor.", "tmdb_id": None},
        {"title": "Pathaan", "genres": "Action|Thriller|Adventure", "overview": "An exiled agent returns to stop a large-scale national threat.", "tmdb_id": None},
        {"title": "Jawan", "genres": "Action|Thriller|Drama", "overview": "A vigilante with a personal mission challenges corruption and power.", "tmdb_id": None},
        {"title": "Drishyam", "genres": "Crime|Thriller|Drama", "overview": "A family man uses sharp planning to protect his loved ones.", "tmdb_id": None},
        {"title": "Bhool Bhulaiyaa", "genres": "Comedy|Horror|Mystery", "overview": "A family estate reveals eerie secrets wrapped in humor and suspense.", "tmdb_id": None},
        {"title": "Border", "genres": "War|Drama|Action", "overview": "Indian soldiers hold a desert outpost during a crucial wartime battle.", "tmdb_id": None},
        {"title": "Sultan", "genres": "Drama|Sport|Action", "overview": "A wrestler seeks redemption in both his career and personal life.", "tmdb_id": None},
        {"title": "Dilwale Dulhania Le Jayenge", "genres": "Romance|Drama", "overview": "A timeless love story battles tradition and family expectations.", "tmdb_id": None},
        {"title": "Shershaah", "genres": "War|Drama|Biography", "overview": "The life and sacrifice of Captain Vikram Batra during the Kargil War.", "tmdb_id": None},

        # More Bollywood
        {"title": "Chak De! India", "genres": "Drama|Sport", "overview": "A disgraced hockey player coaches the national women's team.", "tmdb_id": None},
        {"title": "Munna Bhai M.B.B.S.", "genres": "Comedy|Drama", "overview": "A lovable gangster enrolls in medical college for his father.", "tmdb_id": None},
        {"title": "Sanju", "genres": "Drama|Biography", "overview": "A biographical drama tracing the turbulent life of actor Sanjay Dutt.", "tmdb_id": None},
        {"title": "Bhaag Milkha Bhaag", "genres": "Drama|Sport|Biography", "overview": "The story of sprint legend Milkha Singh and his resilience.", "tmdb_id": None},
        {"title": "Padmaavat", "genres": "Drama|History|Romance", "overview": "A queen, a king, and a ruthless invader collide in medieval India.", "tmdb_id": None},
        {"title": "Krrish", "genres": "Sci-Fi|Action|Adventure", "overview": "A young man discovers extraordinary powers and becomes a superhero.", "tmdb_id": None},
        {"title": "Ra.One", "genres": "Sci-Fi|Action", "overview": "A game villain enters the real world to hunt its creator's son.", "tmdb_id": None},
        {"title": "Don", "genres": "Action|Thriller|Crime", "overview": "A lookalike is pulled into a dangerous underworld deception.", "tmdb_id": None},
        {"title": "Om Shanti Om", "genres": "Romance|Drama|Fantasy", "overview": "A junior artist is reborn to fulfill unfinished love and revenge.", "tmdb_id": None},
        {"title": "Ghajini", "genres": "Action|Thriller|Mystery", "overview": "A man with short-term memory loss hunts his lover's killers.", "tmdb_id": None},
        {"title": "Kal Ho Naa Ho", "genres": "Romance|Drama|Comedy", "overview": "A cheerful stranger transforms the lives of a troubled family.", "tmdb_id": None},
        {"title": "Veer-Zaara", "genres": "Romance|Drama", "overview": "A cross-border love story survives politics and separation.", "tmdb_id": None},
        {"title": "My Name Is Khan", "genres": "Drama|Romance", "overview": "A man with Asperger's undertakes a journey across America after tragedy.", "tmdb_id": None},
        {"title": "Rockstar", "genres": "Drama|Music|Romance", "overview": "An aspiring singer's heartbreak fuels his rise and downfall.", "tmdb_id": None},
        {"title": "Yeh Jawaani Hai Deewani", "genres": "Romance|Drama|Comedy", "overview": "Friends reunite years after a life-changing mountain trip.", "tmdb_id": None},

        # Tollywood (Telugu)
        {"title": "Baahubali: The Beginning", "genres": "Action|Adventure|Drama", "overview": "A young man discovers his royal lineage and a kingdom's past.", "tmdb_id": None},
        {"title": "Baahubali 2: The Conclusion", "genres": "Action|Adventure|Drama", "overview": "The saga of Mahishmati reaches its epic final confrontation.", "tmdb_id": None},
        {"title": "RRR", "genres": "Action|Drama|History", "overview": "Two revolutionaries forge friendship while fighting colonial rule.", "tmdb_id": None},
        {"title": "Pushpa: The Rise", "genres": "Action|Crime|Drama", "overview": "A laborer's rise in the red sandalwood smuggling syndicate.", "tmdb_id": None},
        {"title": "Ala Vaikunthapurramuloo", "genres": "Action|Drama|Comedy", "overview": "A man uncovers the truth about his birth and family identity.", "tmdb_id": None},
        {"title": "Arjun Reddy", "genres": "Romance|Drama", "overview": "A brilliant surgeon spirals after a painful breakup.", "tmdb_id": None},
        {"title": "Maharshi", "genres": "Drama|Action", "overview": "A corporate leader returns to support farmers in his village.", "tmdb_id": None},
        {"title": "Rangasthalam", "genres": "Drama|Action|Thriller", "overview": "A hearing-impaired man battles corruption in a rural village.", "tmdb_id": None},
        {"title": "Pokiri", "genres": "Action|Crime|Thriller", "overview": "An undercover cop infiltrates a powerful crime network.", "tmdb_id": None},
        {"title": "Magadheera", "genres": "Action|Fantasy|Romance", "overview": "Reincarnated lovers confront enemies from a previous life.", "tmdb_id": None},
        {"title": "Eega", "genres": "Fantasy|Action|Comedy", "overview": "A murdered man returns as a fly to seek revenge.", "tmdb_id": None},
        {"title": "Athadu", "genres": "Action|Thriller", "overview": "A hitman assumes a new identity after a political assassination.", "tmdb_id": None},
        {"title": "Jersey", "genres": "Drama|Sport", "overview": "A former cricketer returns to the game for his son.", "tmdb_id": None},
        {"title": "Sye", "genres": "Action|Sport|Drama", "overview": "College students unite through rugby against local goons.", "tmdb_id": None},
        {"title": "Gabbar Singh", "genres": "Action|Comedy", "overview": "A quirky police officer takes on a ruthless criminal.", "tmdb_id": None},

        # Kollywood (Tamil)
        {"title": "Enthiran", "genres": "Sci-Fi|Action|Thriller", "overview": "A humanoid robot turns dangerous after gaining emotions.", "tmdb_id": None},
        {"title": "2.0", "genres": "Sci-Fi|Action", "overview": "A scientist revives his robot creation to tackle a new threat.", "tmdb_id": None},
        {"title": "Vikram", "genres": "Action|Thriller|Crime", "overview": "A black-ops commander investigates murders tied to a drug cartel.", "tmdb_id": None},
        {"title": "Master", "genres": "Action|Thriller|Drama", "overview": "An alcoholic professor confronts a juvenile prison gangster.", "tmdb_id": None},
        {"title": "Kaithi", "genres": "Action|Thriller", "overview": "An ex-con races through one night to save police officers.", "tmdb_id": None},
        {"title": "Ponniyin Selvan: I", "genres": "Drama|History|Adventure", "overview": "A messenger enters Chola politics amid conspiracies and war.", "tmdb_id": None},
        {"title": "Ponniyin Selvan: II", "genres": "Drama|History|Adventure", "overview": "Power struggles intensify as dynastic secrets are revealed.", "tmdb_id": None},
        {"title": "Visaranai", "genres": "Crime|Drama|Thriller", "overview": "A stark tale of custodial violence and institutional abuse.", "tmdb_id": None},
        {"title": "Asuran", "genres": "Action|Drama", "overview": "A farmer fights caste oppression to protect his family.", "tmdb_id": None},
        {"title": "96", "genres": "Romance|Drama", "overview": "Former school sweethearts reconnect at a reunion after decades.", "tmdb_id": None},
        {"title": "Anniyan", "genres": "Action|Thriller|Psychological", "overview": "A man with multiple personalities punishes social corruption.", "tmdb_id": None},
        {"title": "Sivaji", "genres": "Action|Drama|Comedy", "overview": "A software magnate fights black money and systemic corruption.", "tmdb_id": None},
        {"title": "Thuppakki", "genres": "Action|Thriller", "overview": "An army officer dismantles a sleeper-cell terror network.", "tmdb_id": None},
        {"title": "Mersal", "genres": "Action|Thriller|Drama", "overview": "A magician seeks revenge tied to corruption in healthcare.", "tmdb_id": None},
        {"title": "Jai Bhim", "genres": "Drama|Crime", "overview": "A lawyer fights for justice in a landmark custodial case.", "tmdb_id": None},

        # Mollywood (Malayalam)
        {"title": "Drishyam 2", "genres": "Crime|Thriller|Drama", "overview": "The family faces renewed danger years after a hidden crime.", "tmdb_id": None},
        {"title": "Premam", "genres": "Romance|Drama|Comedy", "overview": "A man's life unfolds through three different phases of love.", "tmdb_id": None},
        {"title": "Bangalore Days", "genres": "Drama|Romance|Comedy", "overview": "Three cousins move to Bangalore and discover adulthood.", "tmdb_id": None},
        {"title": "Manichitrathazhu", "genres": "Psychological|Horror|Drama", "overview": "A haunted mansion reveals a layered psychological mystery.", "tmdb_id": None},
        {"title": "Kumbalangi Nights", "genres": "Drama|Family", "overview": "Four estranged brothers rebuild family bonds by the backwaters.", "tmdb_id": None},
        {"title": "Ustad Hotel", "genres": "Drama|Family", "overview": "A young chef learns life lessons through his grandfather's eatery.", "tmdb_id": None},
        {"title": "Charlie", "genres": "Drama|Romance|Adventure", "overview": "A free-spirited artist changes the life of a curious woman.", "tmdb_id": None},
        {"title": "Memories", "genres": "Crime|Thriller|Mystery", "overview": "An alcoholic cop hunts a serial killer with ritual patterns.", "tmdb_id": None},
        {"title": "Mumbai Police", "genres": "Crime|Thriller", "overview": "A police officer investigates a murder linked to his own past.", "tmdb_id": None},
        {"title": "Lucifer", "genres": "Action|Thriller|Crime", "overview": "A mysterious power broker rises after a political leader's death.", "tmdb_id": None},
        {"title": "2018", "genres": "Drama|Thriller|Survival", "overview": "Ordinary people unite during Kerala's devastating floods.", "tmdb_id": None},
        {"title": "Nayattu", "genres": "Crime|Thriller|Drama", "overview": "Three police officers go on the run in a political manhunt.", "tmdb_id": None},
        {"title": "Ayyappanum Koshiyum", "genres": "Drama|Thriller|Action", "overview": "An ego clash between a cop and an ex-soldier escalates dangerously.", "tmdb_id": None},
        {"title": "Minnal Murali", "genres": "Action|Comedy|Fantasy", "overview": "A tailor gains superhero powers after being struck by lightning.", "tmdb_id": None},
        {"title": "Bramayugam", "genres": "Horror|Mystery|Drama", "overview": "A wandering singer enters an ominous mansion with dark secrets.", "tmdb_id": None},
    ]

    return pd.DataFrame(movies)


def main() -> None:
    dataset = build_sample_movies()
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "movies_sample.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(dataset)} movies at: {output_path}")


if __name__ == "__main__":
    main()
