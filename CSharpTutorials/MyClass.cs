﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSharpTutorials
{
    internal class MyClass
    {
        public string myField = string.Empty;

        public MyClass()
        {

        }

        public void MyMethod(int parameter1, string parameter2)
        {
            Console.WriteLine("First Parameter{0}, second parameter {1}", parameter1, parameter2);
        }

        public int MyAutoImplementedProperty { get; set; }

        private int myPropertyVar;
        public int MyPropertyVar
        {
            get { return myPropertyVar; }
            set { myPropertyVar= value; }
        }
    }
}
