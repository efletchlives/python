# pokedex class (manages pokemon that you have)
class Pokedex:
    def __init__ (self):
        # names of possible pokemon
        self.pokemon = ['bulbasaur','charmander','squirtle','pikachu']

        # stores if you have a pokemon in an array (assuming you can only have one of each)
        self.owned = [0] * len(self.pokemon)

    def catch(self,name):
        if name in self.pokemon:
            self.owned[self.pokemon.index(name)] = 1
            print('you caught a',name)
        else:
            print('this is not a pokemon')
    
    def show_pokemon(self):
        print('you have caught:')
        for i, owned in enumerate(self.owned):
            if owned == 1:
                print('\n', self.pokemon[i])

    def all_pokemon(self):
        print('all pokemon:')
        for i in len(self.pokemon):
            status = 'yes' if self.owned[i] == 1 else 'no'
            print(self.pokemon[i],'[',status,']')

    
    # use enumerator to get the name of the pokemon at the stored place

    