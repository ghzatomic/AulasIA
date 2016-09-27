package br.com.caiopaulucci.perceptron;

import java.util.Random;

public class Perceptron{

	private double threshold;

	
	/**
	 * Metodo responsavel por
	 *
	 * entradas = ( Acidez, Fostoro .... )
	 * saidas = ( Valor de cada saida )
	 * threshold = Qu�o perto da linha do grafico
	 * lrate = Learn rate ( A cada itera��o quanto evolui )
	 * epoch = qtde itera��es
	 * 
	 */
	public double[ ] treinar( double[ ][ ] entradas, int[ ] saidas, double threshold, double lrate, int epoch ) {
		double[ ] pesos;
		this.threshold = threshold;
		
		int n = entradas[ 0 ].length;
		int p = saidas.length;
		
		pesos = new double[ n ];
		Random r = new Random();

		//Iniciar todas as conex�es com pesos aleat�rios;
		for ( int i = 0; i < n; i++ ) {
			pesos[ i ] = r.nextDouble();
		}

		//Repita at� que o erro E seja satisfatoriamente pequeno (E <= e) no caso epoch
		for ( int i = 0; i <= epoch; i++ ) {
			int totalError = 0;
			for ( int j = 0; j < p; j++ ) { // Para cada par de treinamento (X,d), fa�a:
				
				int output = output( entradas[ j ],pesos );//Calcular a resposta obtida O;
				int error = saidas[ j ] - output;//Calcular a resposta obtida O;

				totalError += error;
				
				for ( int k = 0; k < n; k++ ) {
					// Calculo do delta para atualizar os pesos ( neta E X )
					double delta = lrate * entradas[ j ][ k ] * error;  
					pesos[ k ] += delta; // Atualizar pesos: Wnovo := W anterior + neta E X
				}

			}
			if ( totalError == 0 ) { //Se nao tem erro pare !
				break;
			}
		}
		return pesos;
	}

	// Testa as saidas !
	public int output( double[ ] input ,double[ ] pesosTreinados ) {
		double sum = 0.0;
		// (Soma ( pesos treinados * dados de entradas )) de todo o treinamento ( dado por dado ou seja .. grafico por grafico )   
		for ( int i = 0; i < input.length; i++ ) {
			sum += pesosTreinados[ i ] * input[ i ];
		}

		if ( sum > threshold ) { // Aqui verifica se esta no nivel aceit�vel !
			return 1;
		} else {
			return 0;
		}
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold( double threshold ) {
		this.threshold = threshold;
	}

}