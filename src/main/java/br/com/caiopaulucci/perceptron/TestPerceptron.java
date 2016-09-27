package br.com.caiopaulucci.perceptron;

public class TestPerceptron{

	public static void main( String[ ] args ) {
		int ideal[] = {
				//0 - Laranja - 1 - Tangerina
				/**/ 0,
		        /**/ 0,
		        /**/ 1,
		        /**/ 1,
		        /**/ 1,
		        /**/ 0,
		        /**/ 1,
		        /**/ 0,
		        /**/ 1,
		        /**/ 1,
		        /**/ 1,
		        /**/ 1,
		        /**/ 1,
		        /**/ 0 };

		//[0] acidez, [1] fosforo
		double data[][] = {
				/**/{ 79.51192437, 20.48807563 },
				/**/{ 92.0660409, 7.933959097 },
				/**/{ 0.642123532, 99.35787647 },
				/**/{ 46.45089699, 53.54910301 },
				/**/{ 6.825159141, 93.17484086 },
				/**/{ 82.03281118, 17.96718882 },
				/**/{ 0.081726617, 99.91827338 },
				/**/{ 66.04740474, 33.95259526 },
				/**/{ 14.10917927, 85.89082073 },
				/**/{ 9.863102261, 90.13689774 },
				/**/{ 42.32055154, 57.67944846 },
				/**/{ 0.959312341, 99.04068766 },
				/**/{ 0.081457561, 99.91854244 },
				/**/{ 72.17355257, 27.82644743 } };

		Perceptron perceptron = new Perceptron();
		double[] treinado = perceptron.treinar(data, ideal, 0.5, 0.05, 100 );

		//----- test -----
		int i = 0;
		for ( double[ ] item : data ) {
			System.out.println("Esperado : "+ ideal[i] +" - Recebido : "+ perceptron.output( item , treinado ) );
			i++;
		}
		//----------------
		double[] exemploTang1 = { 5.2456464, 94.7543536 };
		double[] exemploTang2 = { 2.5654548, 97.4345452 };
		
		double[] exemploLaranja1 = {90.545896,9.454104};
		double[] exemploLaranja2 = {80,20};
		
		//System.out.println(perceptron.output( exemploTang2 , treinado ));

	}

}
