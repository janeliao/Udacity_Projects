import media
import fresh_tomatoes
toy_story=media.Movie('Toy story',
					'A story of a boy and his toys that come to life',
					'https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg',
					'https://www.youtube.com/watch?v=KYz2wyBy3kc')
#print toy_story.storyline
avadar = media.Movie('Avatar',
					'A marine on an alien planet',
					"https://upload.wikimedia.org/wikipedia/en/b/b0/Avatar-Teaser-Poster.jpg",
					'https://www.youtube.com/watch?v=5PSNL1qE6VY')
#print avadar.storyline
#avadar.show_trailer()

ghost_in_the_Shell = media.Movie('Ghost in the Shell',
								"A human saved from a terrible crash, who is cyber-enhanced to be a perfect soldier devoted to stopping the world's most dangerous criminals.",
								"https://upload.wikimedia.org/wikipedia/en/1/11/Ghost_in_the_Shell_%282017_film%29.png",
								'https://www.youtube.com/watch?v=tRkb1X9ovI4')

#print ghost_in_the_Shell.storyline
#ghost_in_the_Shell.show_trailer()
hunger_game = media.Movie('Hunger Game',
						"Each year two young representatives from each district are selected by lottery to participate in The Hunger Games. ",
						'https://upload.wikimedia.org/wikipedia/en/4/42/HungerGamesPoster.jpg',
						'https://www.youtube.com/watch?v=4S9a5V9ODuY')
eight_miles = media.Movie('8 miles',
						"Eminem's memoir film",
						"https://upload.wikimedia.org/wikipedia/en/8/8b/Eight_mile_ver2.jpg",
						"https://www.youtube.com/watch?v=axGVrfwm9L4")
the_avengers = media.Movie('The Avengers',
						'Some super heros save the world',
						'https://upload.wikimedia.org/wikipedia/en/f/f9/TheAvengers2012Poster.jpg',
						'https://www.youtube.com/watch?v=eOrNdBpGMv8')
#the_avengers.show_trailer()
movies = [toy_story,avadar,ghost_in_the_Shell,hunger_game,eight_miles,the_avengers]

fresh_tomatoes.open_movies_page(movies)
#print media.Movie.VALID_RATINGS #VALID_RATINGS IS A VARIABLE NAME
#print media.Movie.__doc__