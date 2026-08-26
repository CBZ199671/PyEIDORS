#include "mfem.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> parse_csv_vector(const std::string &text)
{
   std::vector<double> values;
   std::stringstream stream(text);
   std::string token;
   while (std::getline(stream, token, ','))
   {
      if (token.empty()) { throw std::runtime_error("empty CSV value"); }
      values.push_back(std::stod(token));
   }
   if (values.empty()) { throw std::runtime_error("empty CSV vector"); }
   return values;
}

mfem::DenseMatrix read_currents(const std::string &path, int electrodes)
{
   std::ifstream input(path);
   if (!input) { throw std::runtime_error("cannot open current-pattern file"); }
   std::vector<std::vector<double>> rows;
   std::string line;
   while (std::getline(input, line))
   {
      if (line.empty()) { continue; }
      rows.push_back(parse_csv_vector(line));
   }
   if (static_cast<int>(rows.size()) != electrodes)
   {
      throw std::runtime_error("current-pattern row count mismatch");
   }
   const int drives = static_cast<int>(rows.front().size());
   mfem::DenseMatrix currents(electrodes, drives);
   for (int row = 0; row < electrodes; ++row)
   {
      if (static_cast<int>(rows[row].size()) != drives)
      {
         throw std::runtime_error("ragged current-pattern matrix");
      }
      for (int column = 0; column < drives; ++column)
      {
         currents(row, column) = rows[row][column];
      }
   }
   return currents;
}

mfem::DenseMatrix dense_copy(const mfem::SparseMatrix &matrix)
{
   mfem::DenseMatrix result(matrix.Height(), matrix.Width());
   result = 0.0;
   for (int row = 0; row < matrix.Height(); ++row)
   {
      for (int column = 0; column < matrix.Width(); ++column)
      {
         result(row, column) = matrix(row, column);
      }
   }
   return result;
}

mfem::DenseMatrix helmert_basis(int count)
{
   mfem::DenseMatrix basis(count, count - 1);
   basis = 0.0;
   for (int column = 1; column < count; ++column)
   {
      const double scale = std::sqrt(static_cast<double>(column * (column + 1)));
      for (int row = 0; row < column; ++row)
      {
         basis(row, column - 1) = 1.0 / scale;
      }
      basis(column, column - 1) = -static_cast<double>(column) / scale;
   }
   return basis;
}

void write_matrix(std::ostream &output, const mfem::DenseMatrix &matrix)
{
   output << '[';
   for (int row = 0; row < matrix.Height(); ++row)
   {
      if (row) { output << ','; }
      output << '[';
      for (int column = 0; column < matrix.Width(); ++column)
      {
         if (column) { output << ','; }
         output << matrix(row, column);
      }
      output << ']';
   }
   output << ']';
}

void write_nodes(std::ostream &output, const mfem::Mesh &mesh)
{
   output << '[';
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      if (vertex) { output << ','; }
      const double *coordinate = mesh.GetVertex(vertex);
      output << '[' << coordinate[0] << ',' << coordinate[1] << ']';
   }
   output << ']';
}

void write_cells(std::ostream &output, const mfem::Mesh &mesh)
{
   output << '[';
   mfem::Array<int> vertices;
   for (int element = 0; element < mesh.GetNE(); ++element)
   {
      mesh.GetElementVertices(element, vertices);
      if (vertices.Size() != 3)
      {
         throw std::runtime_error("MFEM imported a non-triangle volume element");
      }
      if (element) { output << ','; }
      output << '[' << vertices[0] << ',' << vertices[1] << ',' << vertices[2]
             << ']';
   }
   output << ']';
}

void write_boundary_edges(std::ostream &output, const mfem::Mesh &mesh,
                          int electrodes)
{
   output << '[';
   mfem::Array<int> vertices;
   for (int element = 0; element < mesh.GetNBE(); ++element)
   {
      mesh.GetBdrElementVertices(element, vertices);
      if (vertices.Size() != 2)
      {
         throw std::runtime_error("MFEM imported a non-segment boundary element");
      }
      const int attribute = mesh.GetBdrAttribute(element);
      const int electrode_label = attribute <= electrodes ? attribute : 0;
      if (element) { output << ','; }
      output << '[' << vertices[0] << ',' << vertices[1] << ','
             << electrode_label << ']';
   }
   output << ']';
}

void ensure_vertex_dof_identity(const mfem::Mesh &mesh,
                                const mfem::FiniteElementSpace &space)
{
   if (space.GetVSize() != mesh.GetNV())
   {
      throw std::runtime_error("P1 scalar DOF count differs from vertex count");
   }
   mfem::Array<int> dofs;
   for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
   {
      space.GetVertexDofs(vertex, dofs);
      if (dofs.Size() != 1 || dofs[0] != vertex)
      {
         throw std::runtime_error("MFEM P1 vertex-to-DOF ordering is not identity");
      }
   }
}

}  // namespace

int main(int argc, char *argv[])
{
   try
   {
      if (argc != 7)
      {
         std::cerr << "usage: mfem_cem_robin MESH OUTPUT FINGERPRINT SIGMA Z_CSV "
                      "CURRENTS_CSV\n";
         return 2;
      }
      const std::string mesh_path = argv[1];
      const std::string output_path = argv[2];
      const std::string fingerprint = argv[3];
      const double conductivity = std::stod(argv[4]);
      const std::vector<double> contact_impedance = parse_csv_vector(argv[5]);
      const int electrodes = static_cast<int>(contact_impedance.size());
      if (!(conductivity > 0.0) || electrodes < 2)
      {
         throw std::runtime_error("conductivity and electrode count are invalid");
      }
      for (const double value : contact_impedance)
      {
         if (!(value > 0.0))
         {
            throw std::runtime_error("contact impedances must be positive");
         }
      }
      mfem::DenseMatrix currents = read_currents(argv[6], electrodes);
      for (int column = 0; column < currents.Width(); ++column)
      {
         double total = 0.0;
         for (int row = 0; row < electrodes; ++row) { total += currents(row, column); }
         if (std::abs(total) > 1.0e-13)
         {
            throw std::runtime_error("current pattern is not balanced");
         }
      }

      mfem::Mesh mesh(mesh_path, 1, 1, true);
      if (mesh.Dimension() != 2 || mesh.SpaceDimension() != 2)
      {
         throw std::runtime_error("MFEM adapter requires a straight 2-D mesh");
      }
      mfem::H1_FECollection collection(1, mesh.Dimension());
      mfem::FiniteElementSpace space(&mesh, &collection, 1, mfem::Ordering::byNODES);
      ensure_vertex_dof_identity(mesh, space);
      const int nodes = space.GetVSize();

      mfem::ConstantCoefficient sigma(conductivity);
      mfem::BilinearForm stiffness_form(&space);
      stiffness_form.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma));
      stiffness_form.Assemble();
      stiffness_form.Finalize();

      const int maximum_boundary_attribute = mesh.bdr_attributes.Max();
      if (maximum_boundary_attribute < electrodes)
      {
         throw std::runtime_error("mesh is missing an electrode boundary attribute");
      }
      mfem::Vector inverse_contact(maximum_boundary_attribute);
      inverse_contact = 0.0;
      for (int electrode = 0; electrode < electrodes; ++electrode)
      {
         inverse_contact(electrode) = 1.0 / contact_impedance[electrode];
      }
      mfem::PWConstCoefficient boundary_coefficient(inverse_contact);
      mfem::BilinearForm boundary_form(&space);
      boundary_form.AddBoundaryIntegrator(
         new mfem::MassIntegrator(boundary_coefficient));
      boundary_form.Assemble();
      boundary_form.Finalize();

      mfem::DenseMatrix coupling(nodes, electrodes);
      mfem::DenseMatrix electrode_matrix(electrodes, electrodes);
      coupling = 0.0;
      electrode_matrix = 0.0;
      for (int electrode = 0; electrode < electrodes; ++electrode)
      {
         mfem::Array<int> marker(maximum_boundary_attribute);
         marker = 0;
         marker[electrode] = 1;
         mfem::ConstantCoefficient inverse_z(1.0 / contact_impedance[electrode]);
         mfem::LinearForm column(&space);
         column.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(inverse_z),
                                      marker);
         column.Assemble();
         double diagonal = 0.0;
         for (int row = 0; row < nodes; ++row)
         {
            coupling(row, electrode) = column[row];
            diagonal += column[row];
         }
         electrode_matrix(electrode, electrode) = diagonal;
      }

      const mfem::SparseMatrix &stiffness_sparse = stiffness_form.SpMat();
      const mfem::SparseMatrix &boundary_sparse = boundary_form.SpMat();
      std::unique_ptr<mfem::SparseMatrix> robin_sparse(
         mfem::Add(stiffness_sparse, boundary_sparse));
      mfem::UMFPackSolver robin_solver(*robin_sparse);
      robin_solver.SetPrintLevel(0);

      mfem::DenseMatrix response(nodes, electrodes);
      for (int electrode = 0; electrode < electrodes; ++electrode)
      {
         mfem::Vector rhs(nodes);
         mfem::Vector solution(nodes);
         for (int row = 0; row < nodes; ++row) { rhs(row) = coupling(row, electrode); }
         robin_solver.Mult(rhs, solution);
         for (int row = 0; row < nodes; ++row)
         {
            response(row, electrode) = solution(row);
         }
      }

      mfem::DenseMatrix transconductance(electrodes, electrodes);
      transconductance = electrode_matrix;
      for (int row = 0; row < electrodes; ++row)
      {
         for (int column = 0; column < electrodes; ++column)
         {
            for (int node = 0; node < nodes; ++node)
            {
               transconductance(row, column) -=
                  coupling(node, row) * response(node, column);
            }
         }
      }

      const mfem::DenseMatrix basis = helmert_basis(electrodes);
      mfem::DenseMatrix reduced(electrodes - 1, electrodes - 1);
      reduced = 0.0;
      for (int row = 0; row < electrodes - 1; ++row)
      {
         for (int column = 0; column < electrodes - 1; ++column)
         {
            for (int left = 0; left < electrodes; ++left)
            {
               for (int right = 0; right < electrodes; ++right)
               {
                  reduced(row, column) += basis(left, row) *
                     transconductance(left, right) * basis(right, column);
               }
            }
         }
      }

      mfem::DenseMatrix coefficients(electrodes - 1, currents.Width());
      coefficients = 0.0;
      for (int row = 0; row < electrodes - 1; ++row)
      {
         for (int column = 0; column < currents.Width(); ++column)
         {
            for (int electrode = 0; electrode < electrodes; ++electrode)
            {
               coefficients(row, column) +=
                  basis(electrode, row) * currents(electrode, column);
            }
         }
      }
      mfem::DenseMatrixInverse reduced_solver(reduced, true);
      reduced_solver.Mult(coefficients);

      mfem::DenseMatrix electrode_voltage(electrodes, currents.Width());
      electrode_voltage = 0.0;
      for (int electrode = 0; electrode < electrodes; ++electrode)
      {
         for (int column = 0; column < currents.Width(); ++column)
         {
            for (int coordinate = 0; coordinate < electrodes - 1; ++coordinate)
            {
               electrode_voltage(electrode, column) +=
                  basis(electrode, coordinate) * coefficients(coordinate, column);
            }
         }
      }

      mfem::DenseMatrix body_potential(nodes, currents.Width());
      for (int column = 0; column < currents.Width(); ++column)
      {
         mfem::Vector rhs(nodes);
         mfem::Vector solution(nodes);
         rhs = 0.0;
         for (int node = 0; node < nodes; ++node)
         {
            for (int electrode = 0; electrode < electrodes; ++electrode)
            {
               rhs(node) += coupling(node, electrode) *
                  electrode_voltage(electrode, column);
            }
         }
         robin_solver.Mult(rhs, solution);
         for (int node = 0; node < nodes; ++node)
         {
            body_potential(node, column) = solution(node);
         }
      }

      const mfem::DenseMatrix stiffness = dense_copy(stiffness_sparse);
      const mfem::DenseMatrix boundary_mass = dense_copy(boundary_sparse);
      const mfem::DenseMatrix robin_matrix = dense_copy(*robin_sparse);
      std::ofstream output(output_path);
      if (!output) { throw std::runtime_error("cannot open output JSON"); }
      output << std::setprecision(17);
      output << "{\"schema\":\"cem-multifem-accuracy-v1\","
                "\"solver\":\"MFEM\","
                "\"formulation\":\"robin_transconductance\","
                "\"implementation\":{\"native_assembly\":true,"
                "\"framework_version\":\"4.9.0\","
                "\"body_solver\":\"UMFPACK\","
                "\"electrode_solver\":\"DenseMatrixInverse\"},"
                "\"discretization\":{\"mesh_fingerprint\":\""
             << fingerprint
             << "\",\"mesh_import_verified\":true,\"potential_order\":1,"
                "\"geometry_order\":1,\"scalar_dtype\":\"float64\","
                "\"imported_nodes\":";
      write_nodes(output, mesh);
      output << ",\"imported_cells_zero_based\":";
      write_cells(output, mesh);
      output << ",\"imported_tagged_boundary_edges_zero_based\":";
      write_boundary_edges(output, mesh, electrodes);
      output << "},\"physical_config\":{\"conductivity\":" << conductivity
             << ",\"contact_impedance\":[";
      for (int electrode = 0; electrode < electrodes; ++electrode)
      {
         if (electrode) { output << ','; }
         output << contact_impedance[electrode];
      }
      output << "],\"currents\":";
      write_matrix(output, currents);
      output << "},\"blocks\":{\"K\":";
      write_matrix(output, stiffness);
      output << ",\"B\":";
      write_matrix(output, boundary_mass);
      output << ",\"C_plus\":";
      write_matrix(output, coupling);
      output << ",\"D\":";
      write_matrix(output, electrode_matrix);
      output << ",\"A_R\":";
      write_matrix(output, robin_matrix);
      output << "},\"solution\":{\"T\":";
      write_matrix(output, transconductance);
      output << ",\"reduced_map\":";
      write_matrix(output, reduced);
      output << ",\"body_potential\":";
      write_matrix(output, body_potential);
      output << ",\"electrode_voltage\":";
      write_matrix(output, electrode_voltage);
      output << "}}\n";
      return 0;
   }
   catch (const std::exception &error)
   {
      std::cerr << "MFEM CEM adapter failed: " << error.what() << '\n';
      return 1;
   }
}
