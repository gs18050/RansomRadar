using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Diagnostics.Tracing.Parsers;
using Microsoft.Diagnostics.Tracing.Etlx;
using Microsoft.Diagnostics.Tracing;
using System.IO;

namespace ConsoleApp2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("please give etl file and target file path.");
                return;
            }
    
            string sourceFileName = args[0];
            string targetFileName = args[1];

            // check if sourceFile exists
            if (!File.Exists(sourceFileName))
            {
                Console.WriteLine($"etl file {sourceFileName} doesn't exist!");
                return;
            }

            try
            {
                using (StreamWriter writer = new StreamWriter(targetFileName))
                {
                    var source = new ETWTraceEventSource(sourceFileName);
                    writer.WriteLine(source.SessionStartTime.ToUniversalTime().ToFileTimeUtc());
                    // Console.WriteLine(source.SessionStartTime.ToUniversalTime().ToFileTimeUtc());
                    // var parser = new DynamicTraceEventParser(source); 

                    //var etlxFile = TraceLog.CreateFromEventTraceLogFile(sourceFileName);
                    //var traceLog = new TraceLog(etlxFile);
                    //Console.WriteLine(traceLog.SessionStartTime.ToUniversalTime().ToFileTimeUtc());
                    // writer.WriteLine(traceLog.SessionStartTime.ToUniversalTime().ToFileTimeUtc());

                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }
    }
}
