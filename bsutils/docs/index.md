# Hello bsutils


## Motivação

O pacote <b>bsutils</b> foi criado para facilitar a implementação de códigos que utilizem a comunicação wifi e a comunicação serial. 

O intuito é facilitar e agilizar o desenvolvimento de algoritmos que necessitem de tais comunicações.


## Desenvolvimento

Este pacote foi desenvolvido sobre a utilização de duas bibliotecas: <b>pyserial</b> e <b>socket</b>.
Como dependência externa temos apenas a biblioteca <b>pyserial</b> que pode ser instalada com o seguinte comando:

	pip install pyserial

Ou 

	conda install pyserial

Para a importação do pacote <b>bsutils</b>, use o seguinte trecho de código:

	from bsutils.board import Wifi   
	from bsutils.board import Serial 


## Classes

Nesta seção será descrita a codfiicação realizada para a construção do algoritmo.

### Serial


Classe responsável pela comunicação serial.

#### __init__()

Construtor, responsável por iniciar algumas variáveis da classe (construtor):


	    def __init__(self):

	        self.rate = 9600
	        self.time = 1
	        self.error = '[ERROR] _'
	        self.OK = '$POK'
	        self.SEND_OK = '$PNEUDOK'




#### open_connection()

Responsável por aplicar permissão na porta serial e abrir a conexão, espera o nome da porta como parâmetro.


	    def open_connection(self, port):

	        try:

	            os.system('sudo chmod -R 777 ' + port)

	            board = serial.Serial(port, baudrate=self.rate, timeout=self.time)

	            return board

	        except serial.SerialException:
	            print(self.error + ' check your parameters and your permission')

	            return None

#### send_message()

Responsável por enviar uma mensagem na porta serial, espera a conexão e a mensagem como parâmetro.


	    def send_message(self, connection, message):

	        if connection is None:
	            print(self.error + ' error in connection')

	            return False
	        else:
	            message = str(message)

	            message = self.create_digit(message.upper())

	            connection.write(message.upper().encode())

	            return True




#### digit_create()

Responsável por criar o checksum da mensagem a ser enviada pela porta serial, espera a mensagem como parâmetro.


    def digit_create(self, information):

        information = information.upper()

        information += ','

        verify_digit = 0

        for digit in str(information):
            verify_digit ^= ord(digit)

        validated_information = ''
        hexadecimal = hex(verify_digit)
        len_hexadecimal = len(hexadecimal)

        if len_hexadecimal == 3:
            validated_information = '0' + hexadecimal[2]
        elif len_hexadecimal == 4:
            validated_information = hexadecimal[2:4]
        else:
            print(self.error + ' unable to generate validation')

        validated_information = information + '*' + validated_information.upper() + '\r\n'

        return validated_information.upper()


#### digit_verify()


Responsável por validar o checksum da mensagem recebida pela porta serial, espera a mensagem como parâmetro.


	  def digit_verify(self, information):

        position = information.find('*')

        result = self.create_digit(information[:position - 1])

        if information[position + 1:] == result[position + 1:]:
            return True
        else:
            return False


### Wifi

Classe responsável pela comunicação wifi.


#### __init__()

Responsável por iniciar algumas variáveis da classe (no momento nenhuma variável foi iniciada):

	def __init__(self):
		pass

#### open_connection()

Responsável por abrir a conexão, espera a porta e o endereço do servidor como parâmetro.

	@staticmethod
    def open_connection(port, host):

        board = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        board.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server = (host, port)
        board.bind(server)
        board.listen(1)

        return board.accept()


## Exemplos


Nesta seção será apresentado algumas aplicações rotineiras com o pacote <b>bsutils</b>.


### Enviando mensagem pela porta serial (pataforma linux):
	
	# importacao da biblioteca
	from bsutils.board import Serial


	def example():

		# objeto da classe
		s = Serial()

		# abrindo a conexao
		connection = s.open_connection(port='/dev/ttyUSB0')

		# enviando uma mensagem
		s.send_message(connection, 'hello world')

		# fechando a conexao
		connection.close()


	if __name__ == '__main__':

		example()


### Realizando a leitura na porta serial
    
    # importação da lib
    from bsutils.board import Serial
    
    def example():
        
        # objeto da classe
        s = Serial()
        
        # abrindo conexao
        connection = s.open_connection(port='/dev/ttyUSB0')
        
        while True:
            
            # verificando a conexao
            if connection is None:
                connection = s.open_connection(port='/dev/ttyUSB0')
    
            try:
                # se possuir comunicação leia e exiba a mensagem
                if connection.inWaiting() > 0:
                    message = connection.readline().decode('utf-8', 'replace')
                    print(message)
            
            # caracter inválido
            except UnicodeError:
                print('Invalid character')
                
            # cabo desconectado
            except IOError:
                print('Cable disconnected')
                
                
    if __name__ == '__main__':
        
        example()
        
        