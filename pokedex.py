class pokedex:
    def __init__ (self):
        # names of possible pokemon
        self.pokemon = ['bulbasaur','charmander','squirtle','pikachu']

        # stores if you have a pokemon in an array (assuming you can only have one of each)
        self.owned = [0] * len(self.pokemon)

    
    # use enumerator to get the name of the pokemon at the stored place

    