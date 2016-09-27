package br.com.caiopaulucci.perceptron;

public class TestPerceptron2Ciencias{

	public static void main( String[ ] args ) {
		int ideal[] = {
				//0 - Musico - 1 - Cientista
				/*Beethoven*/ 0,
				/*Einstein*/ 1,
				/*Mozart*/ 0,
				/*Newton*/ 1,
				/*Tesla*/ 1,
				/*Brahms*/ 0,
				///*Caio*/ 0
				};

		//[0] Entende de musica, [1] Entende de Ciencias
		double data[][] = {
				/**/{ 79.51192437, 20.48807563 },
				/**/{ 10, 90 },
				/**/{ 70, 30 },
				/**/{ 35, 65 },
				/**/{ 5, 95 },
				/**/{ 94, 6 },
				///**/{ 50.5, 49.5 }
				};

		Perceptron perceptron = new Perceptron();
		double[] treinado = perceptron.treinar(data, ideal, 0.6, 0.05, 100 );

		//----- test -----
		/*int i = 0;
		for ( double[ ] item : data ) {
			System.out.println("Esperado : "+ ideal[i] +" - Recebido : "+ perceptron.output( item , treinado ) );
			i++;
		}*/
		//----------------
		double[] exemploMusico = { 51, 49 };
		
		
		System.out.println(perceptron.output( exemploMusico , treinado ));

	}

}
