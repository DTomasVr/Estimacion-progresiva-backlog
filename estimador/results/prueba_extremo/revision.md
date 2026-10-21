# Prueba de extremo a extremo · estado actual

Generador tal cual + estimador con codificadores fp32 | int8. Horas = mediana por tarea; prob = valor probable (rejilla). Banda al 80 %.

## Reservas de limpieza

> Una aplicación web para reservar servicios de limpieza a domicilio, con citas recurrentes y pagos en la app.

**Total** fp32 183.7 h [66.7–363.3] · int8 176.4 h [64.1–348.9] · 37 tareas

### Como miembro de la familia, quiero recibir un pago a la aplicación, para poder alentar mis ganancias.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de pago en la base de datos <br><sub>Store payment details in the database</sub> | 3.53 | 3.59 | 6.0 | 5.0 | [0.8, 15.3] |
| Mostrar los detalles de pago en el panel del miembro de la familia <br><sub>Display payment details on the family member's dashboard</sub> | 3.15 | 3.11 | 3.0 | 3.0 | [0.7, 13.6] |
| Enviar notificaciones de pago al miembro de la familia <br><sub>Send payment notifications to the family member</sub> | 2.71 | 2.55 | 2.5 | 3.0 | [0.6, 11.7] |

### Como enfermera, quiero actualizar la disponibilidad de mis citas recurrentes, para poder asegurar una cobertura adecuada.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear un formulario de actualización de cita recurrente en la aplicación de enfermería <br><sub>Create a recurring appointment update form in the nurse app</sub> | 4.97 | 4.86 | 2.0 | 5.0 | [1.2, 21.5] |
| Enviar una notificación a la enfermera si la solicitud recurrente se actualiza <br><sub>Send a notification to the nurse if the recurring request is updated</sub> | 1.84 | 1.8 | 1.0 | 1.0 | [0.4, 7.9] |
| Actualizar la disponibilidad de las admisiones recurrentes <br><sub>Update the availability of the recurring admissions</sub> | 1.93 | 2.1 | 1.25 | 2.0 | [0.4, 8.3] |
| Actualizar la disponibilidad del planificador de una asignación recurrente <br><sub>Update the scheduler availability of a recurring assignment</sub> | 1.61 | 1.56 | 1.5 | 1.75 | [0.4, 6.9] |
| Permitir que los enfermeros actualicen las citas recurrentes en la base de datos <br><sub>Allow nurses to update recurring appointments in the database</sub> | 2.96 | 3.07 | 3.0 | 3.0 | [0.7, 12.8] |
| Enviar notificaciones a enfermeras sobre recetas recurrentes <br><sub>Send notifications to nurses about recurring prescriptions</sub> | 1.78 | 1.71 | 2.25 | 1.75 | [0.4, 7.7] |
| Proceder para solicitudes recurrentes <br><sub>Procedure for recurring requests</sub> | 2.57 | 2.89 | 2.0 | 2.0 | [0.6, 11.1] |

### Como coordinador de cuidados, quiero hacer un seguimiento de la disponibilidad de citas completas, para poder tomar decisiones informadas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los detalles de la cita en la base de datos <br><sub>Store appointment details in the database</sub> | 3.51 | 3.61 | 3.0 | 3.0 | [0.8, 15.1] |
| Mostrar los detalles de la cita en el panel de coordinación de cuidados <br><sub>Display appointment details on the care coordinator's dashboard</sub> | 3.41 | 3.21 | 3.0 | 3.0 | [0.8, 14.7] |
| Planificar inicialmente las solicitudes completadas <br><sub>Initial planning of completed requests</sub> | 3.66 | 3.88 | 4.0 | 3.0 | [0.8, 15.8] |
| Enviar notificaciones a coordinador de cuidados sobre nuevas citas <br><sub>Send notifications to care coordinators about new appointments</sub> | 2.0 | 1.8 | 4.0 | 3.0 | [0.5, 8.6] |

### Como cuidador, quiero establecer una cita recurrente en la aplicación, para poder planificar mi día de manera eficiente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear un formulario de cita recurrente en la aplicación de cuidador <br><sub>Create a recurring appointment form in the caregiver app</sub> | 5.96 | 6.05 | 6.0 | 4.0 | [1.4, 25.7] |
| Reiniciar el planificador <br><sub>Scheduler restart</sub> | 2.01 | 2.17 | 2.5 | 2.5 | [0.5, 8.7] |
| Proceder a cancelar una aplicación recurrente <br><sub>Procedure to cancel a recurring app</sub> | 1.73 | 1.64 | 1.0 | 1.0 | [0.4, 7.5] |
| Implementar una función de programación de citas recurrentes en la aplicación de cuidador <br><sub>Implement a recurring appointment scheduling feature in the caregiver app</sub> | 7.64 | 7.04 | 4.0 | 7.0 | [1.8, 33.0] |

### Como cuidador, quiero administrar mis registros de visitas, para poder proporcionar mejor atención.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar registros de visitas en el perfil del cuidador <br><sub>Store visit records in the caregiver's profile</sub> | 2.22 | 2.22 | 2.25 | 2.25 | [0.5, 9.6] |
| Mostrar registros de visitas en el panel de la cuidadora <br><sub>Display visit records on the caregiver’s dashboard</sub> | 3.85 | 3.82 | 4.0 | 4.0 | [0.9, 16.6] |
| Eliminar registros <br><sub>Delete records</sub> | 2.42 | 2.21 | 2.0 | 1.75 | [0.6, 10.5] |
| Eliminar registros de búsqueda <br><sub>Delete search records</sub> | 1.58 | 1.58 | 1.75 | 2.0 | [0.4, 6.8] |

### Como coordinador de cuidados, quiero monitorear la disponibilidad de cuidador, para poder asegurar una atención precisa.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar un sistema de notificación para el coordinador de cuidados <br><sub>Implement a notification system for the care coordinator</sub> | 6.07 | 5.86 | 4.0 | 4.0 | [1.4, 26.2] |
| Almacenar los datos de disponibilidad de los cuidadores en la base de datos <br><sub>Store caregiver availability data in the database</sub> | 5.44 | 5.32 | 5.0 | 4.0 | [1.3, 23.5] |
| Añadir capacidad para monitorear colas <br><sub>Add ability to monitor queues</sub> | 6.47 | 5.96 | 6.0 | 6.0 | [1.5, 27.9] |

### Como miembro de la familia, quiero ver una lista de citas recurrentes que se avecinan, para poder planificar mi horario.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Añadir una lista de peticiones recurrentes al planificador <br><sub>Add a list of recurring requests to the scheduler</sub> | 2.85 | 2.92 | 3.0 | 3.0 | [0.7, 12.3] |
| Añadir una asignación adicional recurrente al Scheduler <br><sub>Add an additional recurring assignment to the Scheduler</sub> | 1.58 | 1.67 | 2.5 | 2.5 | [0.4, 6.8] |
| Notificar al familiar si ocurre una repetición <br><sub>Notify the family member if a recurring occurs</sub> | 2.24 | 2.33 | 1.0 | 1.25 | [0.5, 9.6] |

### Como miembro de la familia, quiero ver una lista de las próximas citas, para que pueda planificar mi horario en consecuencia.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar los detalles de la cita en el panel del miembro de la familia <br><sub>Display appointment details on the family member's dashboard</sub> | 3.37 | 3.27 | 3.0 | 2.0 | [0.8, 14.6] |

### Como enfermera, quiero recibir notificaciones sobre las próximas citas, para poder gestionar mis citas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Guardar los detalles de la cita en el perfil de la enfermera. <br><sub>Store appointment details in the nurse's profile</sub> | 2.5 | 2.6 | 2.25 | 2.0 | [0.6, 10.8] |

### Como enfermera, quiero ver mi historial recurrente de citas, para poder rastrear mi disponibilidad.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Añadir documentación para el historial de admisión recurrente <br><sub>Add documentation for recurring admission history</sub> | 2.66 | 2.42 | 3.0 | 2.5 | [0.6, 11.5] |
| Añadir documentación sobre el historial de asignaciones recurrentes <br><sub>Add documentation about recurring assignment history</sub> | 1.47 | 1.33 | 2.0 | 2.0 | [0.3, 6.4] |
| Proporcionar una notificación a la enfermera sobre la historia recurrente de receta <br><sub>Provide a notification to the nurse about recurring prescription history</sub> | 1.57 | 1.61 | 1.25 | 1.5 | [0.4, 6.8] |

### Como miembro de la familia, quiero pedir un recordatorio con mis próximas citas, para que no me pierda la cita.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Enviar un correo electrónico de confirmación al miembro de la familia <br><sub>Send a confirmation email to the family member</sub> | 2.84 | 2.81 | 1.5 | 1.0 | [0.7, 12.2] |
| Almacenar los detalles de la solicitud de recordatorio en la base de datos <br><sub>Store the reminder request details in the database</sub> | 2.52 | 2.37 | 3.0 | 2.5 | [0.6, 10.9] |
| Enviar una notificación por correo electrónico al miembro de la familia sobre el recordatorio <br><sub>Send an email notification to the family member about the reminder</sub> | 2.57 | 2.24 | 2.5 | 2.25 | [0.6, 11.1] |

### Como coordinador de cuidados, quiero seguir el estado de mis citas asignadas, para poder manejar las responsabilidades.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Ser miembro Estado <br><sub>Membership Status</sub> | 2.13 | 2.35 | 1.75 | 1.5 | [0.5, 9.2] |

## Salas de estudio

> Una aplicación móvil donde los estudiantes de una universidad encuentran y reservan salas de estudio en el campus en tiempo real.

**Total** fp32 170.0 h [61.8–336.4] · int8 169.5 h [61.6–335.4] · 32 tareas

### Como estudiante, quiero ver mi disponibilidad a tiempo, para poder planificar mi horario.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de disponibilidad en la base de datos <br><sub>Store the availability data in the database</sub> | 6.23 | 5.99 | 4.5 | 4.5 | [1.4, 26.9] |
| Mostrar la disponibilidad en el panel del estudiante <br><sub>Display the availability on the student's dashboard</sub> | 4.43 | 4.47 | 2.5 | 4.5 | [1.0, 19.1] |
| Enviar una notificación al estudiante sobre la disponibilidad <br><sub>Send a notification to the student about the availability</sub> | 3.07 | 3.04 | 3.0 | 3.0 | [0.7, 13.3] |
| Revisar la disponibilidad a tiempo <br><sub>Please review availability on time</sub> | 1.77 | 1.93 | 2.0 | 2.0 | [0.4, 7.7] |
| Revisar y votar sobre la disponibilidad <br><sub>Please review and vote on availability</sub> | 1.92 | 2.15 | 2.0 | 2.0 | [0.4, 8.3] |

### Como asistente docente, quiero recibir notificaciones cuando un estudiante ha completado sus lecciones, para que pueda prepararme.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar un sistema de notificación en el panel de los auxiliares docentes <br><sub>Implement a notification system in the teaching assistant dashboard</sub> | 6.91 | 6.15 | 3.0 | 3.0 | [1.6, 29.8] |
| Enviar notificaciones a los estudiantes sobre sus lecciones <br><sub>Send notifications to students about their lessons</sub> | 3.71 | 3.52 | 3.5 | 3.5 | [0.9, 16.0] |
| Añadir notificaciones cuando el estudiante haya terminado <br><sub>Add notifications when student is finished</sub> | 2.72 | 2.6 | 2.5 | 2.25 | [0.6, 11.7] |
| Añadir notificación cuando el estudiante termina sus lecciones <br><sub>Add notification when student finishes their lessons</sub> | 3.45 | 3.29 | 3.5 | 3.5 | [0.8, 14.9] |

### Como instructor, quiero seguir la asistencia a los estudiantes, para poder tomar decisiones informadas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de asistencia a los estudiantes en la base de datos <br><sub>Store student attendance data in the database</sub> | 6.02 | 5.88 | 2.5 | 3.5 | [1.4, 26.0] |
| Mostrar la asistencia de los estudiantes en el panel del instructor <br><sub>Display student attendance on the instructor's dashboard</sub> | 3.85 | 4.29 | 2.5 | 3.0 | [0.9, 16.6] |
| Asistir a la admisión semanal <br><sub>Attend weekly admission</sub> | 2.04 | 1.97 | 2.0 | 1.75 | [0.5, 8.8] |

### Como coordinador, quiero supervisar el progreso de la asistencia de los estudiantes, para poder identificar áreas de mejora.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de asistencia a los estudiantes en la base de datos <br><sub>Store student attendance data in the database</sub> | 6.02 | 5.88 | 2.5 | 3.5 | [1.4, 26.0] |

### Como coordinador, quiero gestionar el horario de enseñanza, para poder apoyar el entorno de enseñanza.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear una interfaz de gestión de horarios en el panel de coordinación <br><sub>Create a schedule management interface in the coordinator dashboard</sub> | 7.09 | 6.13 | 7.0 | 6.0 | [1.6, 30.6] |
| Almacenar información sobre el horario de enseñanza en la base de datos <br><sub>Store teaching schedule information in the database</sub> | 2.97 | 2.74 | 4.0 | 3.5 | [0.7, 12.8] |
| Enviar una notificación al coordinador sobre el calendario <br><sub>Send a notification to the coordinator about the schedule</sub> | 2.13 | 2.29 | 2.25 | 2.25 | [0.5, 9.2] |
| Crear una estructura de proyecto separada para el horario de enseñanza <br><sub>Create a separate project structure for the teaching schedule</sub> | 3.6 | 3.96 | 2.0 | 4.0 | [0.8, 15.6] |

### Como instructor, quiero preparar cuestionarios para estudiantes, para que pueda atraer a más profesores.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar las tareas de los cuestionarios en la base de datos <br><sub>Store quiz assignments in the database</sub> | 3.94 | 3.88 | 4.5 | 4.0 | [0.9, 17.0] |
| Mostrar el estado de la asignación del cuestionario en el perfil del instructor <br><sub>Display quiz assignment status on the instructor's profile</sub> | 1.42 | 1.55 | 1.5 | 3.0 | [0.3, 6.1] |
| Notificar a los maestros de nuevas tareas de examen <br><sub>Notify the teachers of new quiz assignments</sub> | 1.84 | 1.84 | 1.75 | 1.75 | [0.4, 7.9] |

### Como asistente docente, quiero ver la asistencia al curso de los estudiantes, para que pueda prepararme para el estudio.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Asistir a la asistencia al curso <br><sub>Attend the course attendance</sub> | 2.75 | 2.85 | 4.5 | 3.0 | [0.6, 11.9] |
| Asistir a la participación semanal <br><sub>Attend the weekly participation</sub> | 2.65 | 2.62 | 1.5 | 1.5 | [0.6, 11.5] |
| Preparar la participación nocturna <br><sub>Prepare the evening participation</sub> | 4.36 | 3.83 | 4.0 | 3.5 | [1.0, 18.8] |

### Como asistente docente, quiero ver estadísticas de progreso de los estudiantes, para poder seguir el progreso.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar estadísticas de progreso en el panel de los asistentes docentes <br><sub>Display progress statistics on the teaching assistant dashboard</sub> | 4.55 | 4.75 | 1.75 | 1.75 | [1.1, 19.6] |

### Como asistente docente, quiero subir las presentaciones de libros, para poder ayudar a los estudiantes académicos.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los detalles de la carga de libros en la base de datos <br><sub>Store book upload details in the database</sub> | 2.89 | 2.54 | 5.0 | 4.5 | [0.7, 12.5] |
| Implementar un formulario de presentación en el panel de los asistentes docentes <br><sub>Implement a submission form in the teaching assistant dashboard</sub> | 5.9 | 5.64 | 3.0 | 3.0 | [1.4, 25.5] |
| Almacenar los detalles de la presentación en la base de datos <br><sub>Store submission details in the database</sub> | 3.76 | 3.64 | 5.0 | 4.5 | [0.9, 16.2] |

### Como estudiante, quiero ver la retroalimentación de los estudiantes, para poder mejorar mi rendimiento.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar comentarios en la página de perfil del estudiante <br><sub>Display feedback on the student's profile page</sub> | 2.26 | 2.3 | 1.5 | 1.5 | [0.5, 9.8] |
| Añadir capacidad para ver la retroalimentación de los estudiantes <br><sub>Add ability to view student feedback</sub> | 3.9 | 4.03 | 4.0 | 4.0 | [0.9, 16.8] |
| Mostrar comentarios en el panel del estudiante <br><sub>Display feedback on the student's dashboard</sub> | 4.27 | 4.26 | 4.5 | 4.5 | [1.0, 18.4] |

### Como instructor, quiero revisar el progreso de los estudiantes, para poder asegurar un desempeño adecuado.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar el progreso del estudiante en el panel <br><sub>Display student progress on the dashboard</sub> | 4.25 | 3.84 | 4.5 | 4.0 | [1.0, 18.3] |

### Como estudiante, quiero proporcionar retroalimentación sobre mi experiencia docente, para poder mejorar mi compromiso.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar datos de retroalimentación en el perfil del estudiante <br><sub>Store feedback data in the student's profile</sub> | 3.71 | 3.6 | 2.0 | 2.0 | [0.9, 16.0] |

## Historias clínicas

> Sistema para que una clínica pequeña gestione historias clínicas, citas y recetas, con acceso por roles para médicos y recepción.

**Total** fp32 206.3 h [74.9–408.2] · int8 194.7 h [70.7–385.3] · 32 tareas

### Como recepcionista, quiero recibir notificaciones para recibir información sobre medicamentos, de modo que pueda comunicarme con los pacientes.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar un formulario de retroalimentación en el perfil del paciente <br><sub>Implement a feedback form in the patient's profile</sub> | 4.44 | 4.25 | 2.0 | 2.25 | [1.0, 19.2] |
| Enviar notificaciones a los pacientes por correo electrónico <br><sub>Send notifications to patients via email</sub> | 4.36 | 4.25 | 4.5 | 4.5 | [1.0, 18.8] |
| Enviar una notificación al paciente sobre el estado <br><sub>Send a notification to the patient about the status</sub> | 3.27 | 2.89 | 3.5 | 3.0 | [0.8, 14.1] |

### Como paciente, quiero acceder a los registros de medicamentos, para que pueda cuidarme.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los registros de pacientes en la base de datos <br><sub>Store patient records in the database</sub> | 5.46 | 4.83 | 4.0 | 5.0 | [1.3, 23.6] |
| Almacenar los registros médicos en la base de datos <br><sub>Store medical records in the database</sub> | 4.68 | 5.08 | 6.0 | 5.0 | [1.1, 20.2] |
| Mostrar los registros médicos en el panel del paciente <br><sub>Display medical records on the patient's dashboard</sub> | 5.28 | 4.86 | 5.5 | 5.0 | [1.2, 22.8] |

### Como médico, quiero actualizar la información del paciente, para poder ofrecer un mejor servicio.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Actualizar documento <br><sub>Update doc</sub> | 1.69 | 1.8 | 1.75 | 1.75 | [0.4, 7.3] |
| Permitir a los médicos actualizar la información del paciente en la base de datos <br><sub>Allow doctors to update patient information in the database</sub> | 3.49 | 3.34 | 5.0 | 3.5 | [0.8, 15.1] |

### Como gerente de clínica, quiero monitorear la asistencia de los pacientes, para poder identificar las tendencias de la clínica.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de asistencia de pacientes en la base de datos <br><sub>Store patient attendance data in the database</sub> | 5.38 | 4.94 | 4.0 | 4.0 | [1.2, 23.2] |
| Mostrar la asistencia de los pacientes en el panel de la clínica <br><sub>Display patient attendance on the clinic dashboard</sub> | 4.95 | 4.71 | 5.0 | 4.5 | [1.1, 21.4] |
| Desarrollar un panel de asistencia a pacientes <br><sub>Develop a patient attendance dashboard</sub> | 10.17 | 10.26 | 5.0 | 5.0 | [2.4, 43.9] |
| Almacenar los datos de asistencia en la base de datos <br><sub>Store attendance data in the database</sub> | 6.04 | 5.92 | 4.0 | 4.0 | [1.4, 26.1] |
| Mostrar los datos de asistencia en el panel del administrador de la clínica <br><sub>Display attendance data on the clinic manager's dashboard</sub> | 5.93 | 5.37 | 6.0 | 5.5 | [1.4, 25.6] |

### Como paciente, quiero seguir el estado de mi cita, para poder manejar mi horario.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar una función de seguimiento de estado en el panel de pacientes <br><sub>Implement a status tracking feature in the patient dashboard</sub> | 8.49 | 8.35 | 9.0 | 5.0 | [2.0, 36.6] |
| Realizar un procedimiento para seguir el estado de un trabajo dado <br><sub>Procedure to track the status of a given job</sub> | 4.29 | 4.1 | 4.5 | 4.0 | [1.0, 18.5] |
| Realizar un procedimiento para controlar el estado de una aplicación <br><sub>Procedure to monitor the status of an app</sub> | 5.31 | 4.73 | 4.0 | 4.0 | [1.2, 22.9] |
| Almacenar el estado de las citas en la base de datos <br><sub>Store the appointment status in the database</sub> | 3.92 | 3.79 | 3.0 | 3.5 | [0.9, 16.9] |

### Como gerente de clínica, quiero administrar el horario del personal para cada paciente, de modo que pueda asegurar follow-up oportuno.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Desarrollar una interfaz de gestión del calendario del personal <br><sub>Develop a staff schedule management interface</sub> | 9.72 | 9.69 | 5.0 | 4.0 | [2.3, 42.0] |
| Almacenar los detalles del horario del personal en la base de datos <br><sub>Store staff schedule details in the database</sub> | 3.16 | 3.44 | 3.5 | 3.5 | [0.7, 13.6] |
| Añadir capacidad para establecer el número de sesiones en el planificador <br><sub>Add ability to set the number of sessions in the scheduler</sub> | 3.26 | 3.19 | 3.5 | 3.0 | [0.8, 14.1] |
| Hacer que el planificador sea más configurable <br><sub>Make the scheduler more configurable</sub> | 3.88 | 3.67 | 3.0 | 4.0 | [0.9, 16.8] |
| Añadir horario basado en el tiempo para el trabajador <br><sub>Add time based scheduling for the worker</sub> | 5.44 | 5.38 | 2.0 | 2.0 | [1.3, 23.5] |

### Como médico, quiero recibir notificaciones sobre las próximas citas, para poder monitorear la programación del médico.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los detalles de la cita en el perfil del médico <br><sub>Store appointment details in the doctor's profile</sub> | 2.28 | 2.48 | 2.0 | 2.0 | [0.5, 9.8] |
| Añadir notificaciones sobre las próximas admisiones <br><sub>Add notifications about upcoming admissions</sub> | 3.66 | 3.3 | 3.0 | 3.0 | [0.8, 15.8] |
| Añadir notificación sobre el estado siguiente <br><sub>Add notification about upcoming status</sub> | 4.01 | 3.69 | 3.0 | 4.0 | [0.9, 17.3] |

### Como paciente, quiero crear un resumen de los registros médicos, para poder programar los horarios de la clínica.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear un formulario de resumen médico en el panel de pacientes <br><sub>Create a medical summary form in the patient dashboard</sub> | 7.81 | 7.22 | 5.0 | 5.0 | [1.8, 33.7] |
| Crear un resumen preliminar de recuperación <br><sub>Create a preliminary recovery summary</sub> | 4.27 | 4.11 | 1.5 | 1.5 | [1.0, 18.4] |

### Como paciente, quiero seguir mis próximas citas, para poder prepararme para el día.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar los detalles de la cita en el panel del paciente <br><sub>Display appointment details on the patient’s dashboard</sub> | 4.12 | 3.45 | 4.0 | 3.0 | [1.0, 17.8] |
| Prepararse para las próximas admisiones <br><sub>Prepare for upcoming admissions</sub> | 5.17 | 4.8 | 3.5 | 4.0 | [1.2, 22.3] |

### Como gerente de clínica, quiero monitorear la disponibilidad de la clínica, para poder manejar al personal de manera efectiva.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de disponibilidad clínica en la base de datos <br><sub>Store clinic availability data in the database</sub> | 5.38 | 5.35 | 5.5 | 3.5 | [1.2, 23.2] |

### Como médico, quiero recibir notificaciones para las citas con pacientes, para poder informar a los pacientes que las causan.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Añadir notificaciones para diagnósticos <br><sub>Add notifications for diagnostics</sub> | 3.04 | 2.72 | 3.0 | 3.0 | [0.7, 13.1] |

### Como recepcionista, quiero actualizar los registros de los pacientes, para poder asegurar un tratamiento adecuado.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Actualizar la documentación sobre la nueva versión <br><sub>Update documentation about the new version</sub> | 2.26 | 2.31 | 2.25 | 2.25 | [0.5, 9.7] |

## Inventario de ferretería

> Un sistema de inventario para una ferretería con varias sucursales, con alertas de stock bajo, transferencias entre tiendas y reportes de ventas.

**Total** fp32 270.7 h [98.3–535.5] · int8 264.7 h [96.1–523.6] · 35 tareas

### Como vendedor, quiero administrar múltiples cuentas de la tienda, para que pueda optimizar la utilización de las acciones.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar cuentas en la base de datos <br><sub>Store accounts in the database</sub> | 4.36 | 4.24 | 6.0 | 6.0 | [1.0, 18.8] |
| Enviar una notificación al vendedor cuando una nueva cuenta está disponible <br><sub>Send a notification to the seller when a new account is available</sub> | 1.87 | 1.62 | 1.75 | 1.75 | [0.4, 8.1] |
| Permitir que se gestionen varias cuentas de almacenamiento <br><sub>Allow multiple store accounts to be managed</sub> | 4.67 | 4.97 | 4.0 | 3.5 | [1.1, 20.1] |
| Añadir capacidad para administrar cuentas de tienda <br><sub>Add ability to manage store accounts</sub> | 6.01 | 5.89 | 4.0 | 4.0 | [1.4, 26.0] |

### Como comprador, quiero comprar una alerta de acciones, para poder comparar precios.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar las alertas de existencias en la base de datos <br><sub>Store stock alerts in the database</sub> | 3.6 | 3.72 | 3.5 | 4.0 | [0.8, 15.5] |
| Mostrar alertas de stock en el panel de compras <br><sub>Display stock alerts on the shopper's dashboard</sub> | 3.72 | 3.73 | 3.5 | 5.0 | [0.9, 16.1] |
| Iniciar cambios en el mercado <br><sub>Initial changes in the marketplace</sub> | 4.99 | 4.83 | 2.5 | 5.0 | [1.2, 21.5] |
| Crear un formulario de alerta de stock en el panel de compras <br><sub>Create a stock alert form in the shopper dashboard</sub> | 5.95 | 5.48 | 2.25 | 5.5 | [1.4, 25.7] |
| Enviar una notificación al comprador sobre la alerta de stock <br><sub>Send a notification to the shopper about the stock alert</sub> | 2.69 | 2.67 | 2.5 | 2.5 | [0.6, 11.6] |

### Como gerente de tienda, quiero monitorear las estadísticas de ventas, para poder analizar el rendimiento del cliente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar estadísticas de ventas en la base de datos <br><sub>Store sales statistics in the database</sub> | 4.77 | 4.78 | 3.0 | 3.0 | [1.1, 20.6] |
| Mostrar estadísticas de ventas en el panel de gerente de tienda <br><sub>Display sales statistics on the manager's dashboard</sub> | 4.57 | 4.47 | 4.5 | 4.5 | [1.1, 19.7] |
| Añadir estadísticas al gestor de la tienda <br><sub>Add statistics to the store manager</sub> | 6.69 | 6.42 | 3.0 | 3.0 | [1.6, 28.9] |

### Como recolector de almacenes, quiero recibir notificaciones sobre el proceso de recogida inmediata, para que pueda gestionar la recogida de manera efectiva.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar un sistema de notificación en el panel recolector de almacenes <br><sub>Implement a notification system in the warehouse picker dashboard</sub> | 7.58 | 8.27 | 8.0 | 8.0 | [1.8, 32.7] |
| Enviar notificaciones al operador de almacén cuando se procesa un proceso de recolección <br><sub>Send notifications to the warehouse operator when a picking process is processed</sub> | 2.98 | 2.98 | 3.0 | 3.0 | [0.7, 12.8] |
| Añadir notificación sobre el proceso de selección inmediata <br><sub>Add notification about immediate picking process</sub> | 3.21 | 3.02 | 3.0 | 3.0 | [0.7, 13.8] |

### Como vendedor, quiero configurar informes de retorno, para que pueda gestionar los horarios de entrega de manera eficiente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los informes de retorno en la base de datos <br><sub>Store return reports in the database</sub> | 3.44 | 2.73 | 4.0 | 4.0 | [0.8, 14.8] |
| Mostrar informes de retorno en el panel del vendedor <br><sub>Display return reports on the seller's dashboard</sub> | 3.9 | 3.62 | 4.0 | 3.5 | [0.9, 16.8] |
| Crear una lista de los informes disponibles <br><sub>Create a list of the available reports</sub> | 3.53 | 3.67 | 3.0 | 4.0 | [0.8, 15.2] |

### Como gerente de tienda, quiero seguir las tendencias de ventas, para poder identificar tendencias y patrones futuros.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar las tendencias de ventas en la base de datos <br><sub>Store sales trends in the database</sub> | 5.25 | 5.06 | 3.0 | 3.0 | [1.2, 22.7] |
| Mostrar las tendencias de ventas en el panel del administrador de tiendas <br><sub>Display sales trends on the store manager dashboard</sub> | 4.76 | 4.58 | 5.0 | 4.5 | [1.1, 20.6] |
| Generar informes sobre las tendencias de ventas <br><sub>Generate reports on sales trends</sub> | 5.58 | 4.9 | 2.5 | 3.5 | [1.3, 24.1] |

### Como vendedor, quiero administrar múltiples tiendas para un lote, de modo que pueda manejar múltiples ofertas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Guardar la gestión de lotes en la base de datos <br><sub>Store the batch management in the database</sub> | 5.24 | 5.22 | 5.0 | 6.0 | [1.2, 22.6] |
| Permitir que varios almacenes sean gestionados por un lote <br><sub>Allow multiple stores to be managed by one batch</sub> | 6.0 | 5.81 | 5.0 | 4.0 | [1.4, 25.9] |

### Como gerente de tienda, quiero monitorear las estadísticas de ventas, para poder optimizar los recursos.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar estadísticas de ventas en la base de datos <br><sub>Store sales statistics in the database</sub> | 4.77 | 4.78 | 3.0 | 3.0 | [1.1, 20.6] |

### Como gerente de tienda, quiero monitorear los niveles de existencias, para poder optimizar el inventario.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los niveles de existencias en la base de datos <br><sub>Store stock levels in the database</sub> | 4.43 | 4.05 | 3.0 | 4.0 | [1.0, 19.1] |
| Mostrar los niveles de stock en el panel de gerente de tienda <br><sub>Display stock levels on the manager's dashboard</sub> | 4.48 | 4.37 | 6.0 | 4.5 | [1.0, 19.3] |
| Añadir documentación para el administrador de la tienda <br><sub>Add documentation for the store manager</sub> | 4.32 | 3.87 | 5.0 | 4.0 | [1.0, 18.7] |
| Implementar una función de seguimiento de niveles de stock en el panel del administrador de tiendas <br><sub>Implement a stock levels tracking feature in the store manager dashboard</sub> | 10.03 | 10.01 | 5.0 | 10.0 | [2.3, 43.3] |

### Como gerente de tienda, quiero monitorear el volumen de inventario de los niveles de existencias, para poder rastrear los niveles de existencias.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Añadir niveles de inventario al administrador de la tienda <br><sub>Add inventory levels to the store manager</sub> | 4.35 | 3.68 | 4.0 | 3.5 | [1.0, 18.8] |
| Almacenar los datos del volumen de inventario en la base de datos <br><sub>Store inventory volume data in the database</sub> | 5.25 | 5.12 | 6.0 | 4.0 | [1.2, 22.7] |
| Generar alertas para los niveles de existencias <br><sub>Generate alerts for stock levels</sub> | 3.32 | 3.65 | 7.0 | 7.0 | [0.8, 14.3] |

### Como vendedor, quiero establecer un real-time informes de precios, para que pueda ofrecer un mejor servicio al cliente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar los datos real-time en el panel del vendedor <br><sub>Display real-time data on the seller's dashboard</sub> | 6.64 | 6.26 | 7.0 | 6.0 | [1.5, 28.7] |
| Añadir informes de precios <br><sub>Add pricing reports</sub> | 3.7 | 3.59 | 3.0 | 4.5 | [0.9, 16.0] |
| Añadir un informe adicional sobre precios <br><sub>Add an additional pricing report</sub> | 2.31 | 2.27 | 3.5 | 3.5 | [0.5, 10.0] |

### Como gerente de tienda, quiero revisar los niveles de stock de los productos, para que pueda gestionar el inventario de manera eficiente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Añadir documentación para los nuevos niveles de inventario <br><sub>Add documentation for the new inventory levels</sub> | 2.34 | 2.15 | 3.0 | 3.0 | [0.5, 10.1] |

## Bolsa de trabajo

> Portal de empleo para egresados de una universidad, donde las empresas publican ofertas y los egresados postulan con su CV.

**Total** fp32 177.7 h [64.5–351.5] · int8 181.4 h [65.9–358.8] · 28 tareas

### Como reclutador, quiero crear una lista de trabajo personalizada, para poder rastrear su progreso.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear un formulario de listado de trabajos en el panel de reclutamiento <br><sub>Create a job listing form in the recruiter dashboard</sub> | 5.58 | 5.8 | 3.0 | 2.0 | [1.3, 24.1] |
| Almacenar los detalles de la lista de tareas en la base de datos <br><sub>Store job listing details in the database</sub> | 3.56 | 3.53 | 3.5 | 3.5 | [0.8, 15.4] |
| Enviar una notificación al reclutador sobre la nueva lista de trabajo <br><sub>Send a notification to the recruiter about the new job listing</sub> | 1.66 | 1.89 | 3.0 | 2.5 | [0.4, 7.2] |

### Como candidato, quiero ver la tesis general de mi trabajo, para poder tomar decisiones informadas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar el estado en el panel del candidato <br><sub>Display the status on the candidate’s dashboard</sub> | 3.56 | 3.46 | 1.5 | 3.0 | [0.8, 15.4] |

### Como administrador de HR, quiero monitorear el número de trabajos completados, para que pueda administrar los recursos de manera eficiente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar trabajos completados en la base de datos <br><sub>Store completed jobs in the database</sub> | 4.71 | 4.37 | 4.0 | 3.0 | [1.1, 20.3] |

### Como gerente de contratación, quiero gestionar el acceso a los resultados del candidato, para que pueda optimizar futuras aplicaciones.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los resultados en la base de datos <br><sub>Store the results in the database</sub> | 5.22 | 5.29 | 4.0 | 5.0 | [1.2, 22.5] |
| Mostrar los resultados en el panel del candidato <br><sub>Display the results on the candidate’s dashboard</sub> | 5.29 | 5.37 | 5.5 | 5.5 | [1.2, 22.8] |
| Notificar al candidato los resultados <br><sub>Notify the candidate of the results</sub> | 2.9 | 2.79 | 3.0 | 3.0 | [0.7, 12.5] |

### Como candidato, quiero recibir notificaciones sobre las próximas asignaciones, para poder planificar mi día en consecuencia.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear un sistema de notificación para las próximas asignaciones <br><sub>Create a notification system for upcoming assignments</sub> | 7.34 | 7.87 | 7.0 | 8.0 | [1.7, 31.7] |
| Añadir notificación sobre las próximas asignaciones <br><sub>Add notification about upcoming assignments</sub> | 3.52 | 3.57 | 3.5 | 3.5 | [0.8, 15.2] |

### Como reclutador, quiero seguir el estado de un trabajo, para poder manejar el proceso de reclutamiento adecuado.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar el estado de los trabajos en la base de datos <br><sub>Store job status in the database</sub> | 3.64 | 3.72 | 2.0 | 2.0 | [0.8, 15.7] |

### Como administrador de HR, quiero generar informes sobre reclutamiento y compromiso, para poder revisar el proceso de reclutamiento.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Generar informes sobre rh <br><sub>generate reports on rh</sub> | 3.54 | 3.44 | 5.0 | 4.5 | [0.8, 15.3] |

### Como administrador de HR, quiero monitorear el número de posiciones enviadas, para poder analizar tendencias y tareas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear panel para las posiciones presentadas <br><sub>Create dashboard for submitted positions</sub> | 7.09 | 7.06 | 7.0 | 7.0 | [1.6, 30.6] |

### Como administrador de HR, quiero generar informes sobre las calificaciones pendientes, para que pueda identificar tendencias.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Generar informe sobre la categoría pendiente <br><sub>generate report on pending grade</sub> | 2.37 | 2.84 | 1.25 | 1.75 | [0.6, 10.2] |
| Almacenar los informes en la base de datos <br><sub>Store reports in the database</sub> | 3.63 | 3.21 | 4.0 | 4.0 | [0.8, 15.7] |
| Mostrar informes sobre grados pendientes <br><sub>Display reports on pending grades</sub> | 2.95 | 2.88 | 1.75 | 1.75 | [0.7, 12.7] |

### Como candidato, quiero ver el estado de mi enseñanza, para poder evaluar mi trabajo.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Revisar el estado de mi enseñanza <br><sub>Please review the status of my teaching</sub> | 2.01 | 1.83 | 2.0 | 1.75 | [0.5, 8.7] |
| Revisar y discuta el estado de su enseñanza <br><sub>Please review and discuss the status of your teaching</sub> | 2.11 | 2.4 | 2.0 | 1.75 | [0.5, 9.1] |

### Como reclutador, quiero acceder al estado de reclutamiento de un candidato, para poder adaptarme a los reclutadores.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Recuperar el estado del candidato de la base de datos <br><sub>Retrieve the candidate's status from the database</sub> | 4.56 | 4.21 | 1.0 | 4.0 | [1.1, 19.7] |
| Almacenar los datos de estado en el perfil del candidato <br><sub>Store the status data in the candidate's profile</sub> | 4.81 | 4.84 | 3.0 | 5.0 | [1.1, 20.8] |
| Almacenar el estado de los candidatos en la base de datos <br><sub>Store candidate status in the database</sub> | 4.44 | 4.0 | 4.0 | 4.0 | [1.0, 19.2] |
| Enviar notificaciones a los reclutadores sobre su estado <br><sub>Send notifications to recruiters about their status</sub> | 3.28 | 3.0 | 3.5 | 3.0 | [0.8, 14.1] |

### Como candidato, quiero recibir notificaciones para las próximas tareas, para poder priorizar el plazo.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Enviar una notificación al candidato sobre el estado <br><sub>Send a notification to the candidate about the status</sub> | 2.73 | 2.8 | 2.5 | 3.0 | [0.6, 11.8] |
| Enviar notificaciones al candidato por correo electrónico <br><sub>Send notifications to the candidate via email</sub> | 4.68 | 4.81 | 3.0 | 5.0 | [1.1, 20.2] |
| Añadir notificaciones sobre las próximas tareas <br><sub>Add notifications about upcoming tasks</sub> | 3.41 | 3.21 | 2.5 | 2.5 | [0.8, 14.7] |
| Crear UI para el estado de envío <br><sub>Create UI for the submitted status</sub> | 5.2 | 5.44 | 2.0 | 2.0 | [1.2, 22.5] |
| Aplicar un sistema de notificación para las presentaciones de candidatos <br><sub>Implement a notification system for candidate submissions</sub> | 7.29 | 6.86 | 4.0 | 7.0 | [1.7, 31.4] |
| Añadir notificación para Calendario <br><sub>Add notification for Calendar</sub> | 3.12 | 2.74 | 3.0 | 3.0 | [0.7, 13.5] |

## Delivery app (en)

> A food delivery app for a small city, where restaurants publish menus, customers order and pay online, and couriers get routes.

**Total** fp32 211.5 h [76.8–418.5] · int8 201.2 h [73.1–398.0] · 35 tareas

### Como mensajero, quiero establecer una ruta de entrega para mi restaurante, para que pueda gestionar mi entrega de manera eficiente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar una función de configuración de la ruta de entrega en la aplicación de mensajería <br><sub>Implement a delivery route configuration feature in the courier app</sub> | 7.06 | 6.78 | 4.0 | 4.0 | [1.6, 30.4] |
| Almacenar los detalles de la ruta de entrega en la base de datos <br><sub>Store delivery route details in the database</sub> | 3.72 | 4.29 | 5.0 | 5.0 | [0.9, 16.0] |
| Enviar una notificación al mensajero sobre la ruta de entrega <br><sub>Send a notification to the courier about the delivery route</sub> | 2.24 | 2.33 | 2.25 | 2.25 | [0.5, 9.7] |
| Añadir documentación para la ruta de entrega <br><sub>Add documentation for the delivery route</sub> | 2.65 | 2.48 | 4.0 | 3.0 | [0.6, 11.4] |
| Crear un formulario de configuración de la ruta de entrega en la aplicación de mensajería <br><sub>Create a delivery route setup form in the courier app</sub> | 6.27 | 5.56 | 4.5 | 5.5 | [1.5, 27.1] |
| Mostrar los detalles de la ruta de entrega en el panel de mensajero <br><sub>Display delivery route details on the courier's dashboard</sub> | 3.9 | 3.49 | 3.0 | 3.5 | [0.9, 16.8] |
| Notificar al mensajero si la ruta está disponible <br><sub>Notify the courier if the route is available</sub> | 1.76 | 1.82 | 2.0 | 3.0 | [0.4, 7.6] |

### Como cliente, quiero guardar mis suministros en línea, para que pueda conocer mis pedidos.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar suministros en el perfil del cliente <br><sub>Store supplies in the customer's profile</sub> | 3.64 | 3.27 | 4.0 | 3.5 | [0.8, 15.7] |
| Enviar un correo electrónico de confirmación al cliente cuando haya suministros disponibles <br><sub>Send a confirmation email to the customer when supplies are available</sub> | 2.89 | 3.07 | 2.25 | 2.25 | [0.7, 12.5] |
| Almacenar suministros en la base de datos <br><sub>Store supplies in the database</sub> | 4.47 | 4.14 | 4.5 | 5.0 | [1.0, 19.3] |
| Mostrar los suministros en el panel del cliente <br><sub>Display supplies on the customer's dashboard</sub> | 4.76 | 4.42 | 5.0 | 4.5 | [1.1, 20.6] |

### Como propietario de un restaurante, quiero gestionar múltiples mesas de los elementos del menú, para que pueda ofrecer diferentes pedidos de entrega al restaurante.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Desarrollar una interfaz de gestión de elementos de menú <br><sub>Develop a menu items management interface</sub> | 8.14 | 7.57 | 8.0 | 8.0 | [1.9, 35.1] |
| Almacenar los elementos del menú en la base de datos <br><sub>Store the menu items in the database</sub> | 4.34 | 4.15 | 4.5 | 4.0 | [1.0, 18.7] |
| Añadir tablas para los elementos del menú <br><sub>Add tables for the menu items</sub> | 3.17 | 3.32 | 2.0 | 1.5 | [0.7, 13.7] |
| Almacenar las tablas en la base de datos <br><sub>Store the tables in the database</sub> | 4.41 | 4.01 | 4.5 | 4.0 | [1.0, 19.0] |
| Mostrar las mesas en el panel del restaurante <br><sub>Display the tables on the restaurant dashboard</sub> | 3.57 | 3.66 | 3.5 | 3.5 | [0.8, 15.4] |

### Como gerente de operaciones, quiero monitorear la ubicación de entrega de pedidos, para que pueda optimizar la entrega.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los datos de la ubicación de la entrega en la base de datos <br><sub>Store delivery location data in the database</sub> | 4.22 | 4.04 | 5.0 | 3.0 | [1.0, 18.2] |
| Mostrar la ubicación de la entrega en el panel de pedidos <br><sub>Display delivery location on the order dashboard</sub> | 2.12 | 2.06 | 1.5 | 2.0 | [0.5, 9.1] |
| Añadir documentación para el lugar de entrega de los pedidos <br><sub>Add documentation for the delivery location of orders</sub> | 2.0 | 1.94 | 4.0 | 4.0 | [0.5, 8.6] |
| Almacenar los datos de ubicación del pedido en la base de datos <br><sub>Store order location data in the database</sub> | 3.81 | 3.76 | 2.5 | 2.25 | [0.9, 16.4] |

### Como cliente, quiero recibir notificaciones cuando el pedido esté listo, para poder planificar mi horario.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Implementar un sistema de notificación en el panel de clientes <br><sub>Implement a notification system in the customer dashboard</sub> | 10.38 | 9.42 | 4.5 | 5.0 | [2.4, 44.8] |

### Como mensajero, quiero rastrear el tiempo de entrega que paso en la recogida, para poder planificar mi ruta.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Mostrar el tiempo de entrega empleado en la aplicación del mensajero <br><sub>Display the delivery time spent on the courier's app</sub> | 3.9 | 3.6 | 4.0 | 3.5 | [0.9, 16.8] |

### Como mensajero, quiero establecer una ruta de entrega, para que pueda gestionar las entregas de manera más eficiente.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Crear una función de programación de rutas en la aplicación de mensajería <br><sub>Create a route scheduling feature in the courier app</sub> | 7.04 | 6.75 | 3.0 | 4.0 | [1.6, 30.4] |
| Almacenar los detalles de la ruta en la base de datos <br><sub>Store route details in the database</sub> | 3.73 | 3.78 | 5.0 | 5.0 | [0.9, 16.1] |

### Como gerente de operaciones, quiero supervisar el estado de entrega de los pedidos, para que pueda asegurar la entrega oportuna.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Añadir capacidad para supervisar el estado de entrega de los pedidos <br><sub>Add ability to monitor the delivery status of orders</sub> | 5.57 | 4.82 | 5.5 | 5.0 | [1.3, 24.0] |
| Almacenar el estado del pedido en la base de datos <br><sub>Store order status in the database</sub> | 3.03 | 2.98 | 2.5 | 2.0 | [0.7, 13.1] |
| Mostrar el estado del pedido en el panel del sistema <br><sub>Display order status on the system dashboard</sub> | 3.04 | 2.7 | 3.0 | 3.0 | [0.7, 13.1] |
| Notificar al almacén gerente de operaciones el estado del pedido <br><sub>Notify the warehouse manager of the order status</sub> | 2.13 | 2.27 | 3.0 | 2.25 | [0.5, 9.2] |

### Como mensajero, quiero seleccionar un pedido pick-in para una entrega, de modo que pueda elegir la mejor opción para la entrega.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los detalles de entrega de pick-in en la base de datos <br><sub>Store pick-in delivery details in the database</sub> | 3.85 | 3.93 | 4.0 | 4.0 | [0.9, 16.6] |
| Añadir un pedido pick-in a la página de entrega <br><sub>Add a pick-in order to the delivery page</sub> | 2.98 | 2.89 | 3.0 | 2.0 | [0.7, 12.9] |

### Como propietario de un restaurante, quiero gestionar los tiempos de entrega, para que pueda optimizar la disponibilidad.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Almacenar los tiempos de entrega en la base de datos <br><sub>Store delivery times in the database</sub> | 3.5 | 3.45 | 2.0 | 3.0 | [0.8, 15.1] |

### Como propietario de un restaurante, quiero gestionar el inventario de pedidos diarios, para que pueda tomar decisiones informadas.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Desarrollar una interfaz diaria de gestión de pedidos <br><sub>Develop a daily order management interface</sub> | 7.66 | 6.99 | 2.5 | 2.5 | [1.8, 33.1] |
| Almacenar detalles de pedidos diarios en la base de datos <br><sub>Store daily order details in the database</sub> | 3.97 | 3.95 | 2.5 | 2.5 | [0.9, 17.1] |
| Mostrar detalles de pedido diario en el panel del propietario del restaurante <br><sub>Display daily order details on the restaurant owner dashboard</sub> | 3.17 | 3.07 | 3.0 | 2.5 | [0.7, 13.7] |

### Como propietario de un restaurante, quiero saber cuándo es el momento de el pedido, para que pueda planificar en consecuencia.

| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |
|---|---|---|---|---|---|
| Cuando el pedido se debe a un pedido adicional, me gustaría poder elegir qué pedidos se deben a el pedido determinada <br><sub>When the ordering is due to an additional order, I would like to be able to pick which orders are due to the certain order</sub> | 3.02 | 3.09 | 3.0 | 3.0 | [0.7, 13.0] |
