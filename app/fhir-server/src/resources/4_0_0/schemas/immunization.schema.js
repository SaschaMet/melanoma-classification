const {
	GraphQLNonNull,
	GraphQLEnumType,
	GraphQLList,
	GraphQLUnionType,
	GraphQLString,
	GraphQLBoolean,
	GraphQLObjectType,
} = require('graphql');
const IdScalar = require('../scalars/id.scalar.js');
const UriScalar = require('../scalars/uri.scalar.js');
const CodeScalar = require('../scalars/code.scalar.js');
const DateTimeScalar = require('../scalars/datetime.scalar.js');
const DateScalar = require('../scalars/date.scalar.js');

/**
 * @name exports
 * @summary Immunization Schema
 */
module.exports = new GraphQLObjectType({
	name: 'Immunization',
	description:
		'Describes the event of a patient being administered a vaccine or a record of an immunization as reported by a patient, a clinician or another party.',
	fields: () => ({
		resourceType: {
			type: new GraphQLNonNull(
				new GraphQLEnumType({
					name: 'Immunization_Enum_schema',
					values: { Immunization: { value: 'Immunization' } },
				}),
			),
			description: 'Type of resource',
		},
		_id: {
			type: require('./element.schema.js'),
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		id: {
			type: IdScalar,
			description:
				'The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.',
		},
		meta: {
			type: require('./meta.schema.js'),
			description:
				'The metadata about the resource. This is content that is maintained by the infrastructure. Changes to the content might not always be associated with version changes to the resource.',
		},
		_implicitRules: {
			type: require('./element.schema.js'),
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		implicitRules: {
			type: UriScalar,
			description:
				'A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.',
		},
		_language: {
			type: require('./element.schema.js'),
			description: 'The base language in which the resource is written.',
		},
		language: {
			type: CodeScalar,
			description: 'The base language in which the resource is written.',
		},
		text: {
			type: require('./narrative.schema.js'),
			description:
				"A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human. The narrative need not encode all the structured data, but is required to contain sufficient detail to make it 'clinically safe' for a human to just read the narrative. Resource definitions may define what content should be represented in the narrative to ensure clinical safety.",
		},
		contained: {
			type: new GraphQLList(require('./resourcelist.schema')),
			description:
				'These resources do not have an independent existence apart from the resource that contains them - they cannot be identified independently, and nor can they have their own independent transaction scope.',
		},
		extension: {
			type: new GraphQLList(require('./extension.schema.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the resource. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		modifierExtension: {
			type: new GraphQLList(require('./extension.schema.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the resource and that modifies the understanding of the element that contains it and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer is allowed to define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		identifier: {
			type: new GraphQLList(require('./identifier.schema.js')),
			description: 'A unique identifier assigned to this immunization record.',
		},
		_status: {
			type: require('./element.schema.js'),
			description: 'Indicates the current status of the immunization event.',
		},
		status: {
			type: new GraphQLNonNull(CodeScalar),
			description: 'Indicates the current status of the immunization event.',
		},
		statusReason: {
			type: require('./codeableconcept.schema.js'),
			description:
				'Indicates the reason the immunization event was not performed.',
		},
		vaccineCode: {
			type: new GraphQLNonNull(require('./codeableconcept.schema.js')),
			description: 'Vaccine that was administered or was to be administered.',
		},
		patient: {
			type: new GraphQLNonNull(
				new GraphQLUnionType({
					name: 'Immunizationpatient_patient_Union',
					description:
						'The patient who either received or did not receive the immunization.',
					types: () => [require('./patient.schema.js')],
					resolveType(data) {
						if (data && data.resourceType === 'Patient') {
							return require('./patient.schema.js');
						}
					},
				}),
			),
			description:
				'The patient who either received or did not receive the immunization.',
		},
		encounter: {
			type: new GraphQLUnionType({
				name: 'Immunizationencounter_encounter_Union',
				description:
					'The visit or admission or other contact between patient and health care provider the immunization was performed as part of.',
				types: () => [require('./encounter.schema.js')],
				resolveType(data) {
					if (data && data.resourceType === 'Encounter') {
						return require('./encounter.schema.js');
					}
				},
			}),
			description:
				'The visit or admission or other contact between patient and health care provider the immunization was performed as part of.',
		},
		_occurrenceDateTime: {
			type: require('./element.schema.js'),
			description: 'Date vaccine administered or was to be administered.',
		},
		occurrenceDateTime: {
			type: new GraphQLNonNull(DateTimeScalar),
			description: 'Date vaccine administered or was to be administered.',
		},
		_occurrenceString: {
			type: require('./element.schema.js'),
			description: 'Date vaccine administered or was to be administered.',
		},
		occurrenceString: {
			type: new GraphQLNonNull(GraphQLString),
			description: 'Date vaccine administered or was to be administered.',
		},
		_recorded: {
			type: require('./element.schema.js'),
			description:
				'The date the occurrence of the immunization was first captured in the record - potentially significantly after the occurrence of the event.',
		},
		recorded: {
			type: DateTimeScalar,
			description:
				'The date the occurrence of the immunization was first captured in the record - potentially significantly after the occurrence of the event.',
		},
		_primarySource: {
			type: require('./element.schema.js'),
			description:
				'An indication that the content of the record is based on information from the person who administered the vaccine. This reflects the context under which the data was originally recorded.',
		},
		primarySource: {
			type: GraphQLBoolean,
			description:
				'An indication that the content of the record is based on information from the person who administered the vaccine. This reflects the context under which the data was originally recorded.',
		},
		reportOrigin: {
			type: require('./codeableconcept.schema.js'),
			description:
				'The source of the data when the report of the immunization event is not based on information from the person who administered the vaccine.',
		},
		location: {
			type: new GraphQLUnionType({
				name: 'Immunizationlocation_location_Union',
				description:
					'The service delivery location where the vaccine administration occurred.',
				types: () => [require('./location.schema.js')],
				resolveType(data) {
					if (data && data.resourceType === 'Location') {
						return require('./location.schema.js');
					}
				},
			}),
			description:
				'The service delivery location where the vaccine administration occurred.',
		},
		manufacturer: {
			type: new GraphQLUnionType({
				name: 'Immunizationmanufacturer_manufacturer_Union',
				description: 'Name of vaccine manufacturer.',
				types: () => [require('./organization.schema.js')],
				resolveType(data) {
					if (data && data.resourceType === 'Organization') {
						return require('./organization.schema.js');
					}
				},
			}),
			description: 'Name of vaccine manufacturer.',
		},
		_lotNumber: {
			type: require('./element.schema.js'),
			description: 'Lot number of the  vaccine product.',
		},
		lotNumber: {
			type: GraphQLString,
			description: 'Lot number of the  vaccine product.',
		},
		_expirationDate: {
			type: require('./element.schema.js'),
			description: 'Date vaccine batch expires.',
		},
		expirationDate: {
			type: DateScalar,
			description: 'Date vaccine batch expires.',
		},
		site: {
			type: require('./codeableconcept.schema.js'),
			description: 'Body site where vaccine was administered.',
		},
		route: {
			type: require('./codeableconcept.schema.js'),
			description:
				'The path by which the vaccine product is taken into the body.',
		},
		doseQuantity: {
			type: require('./quantity.schema.js'),
			description: 'The quantity of vaccine product that was administered.',
		},
		performer: {
			type: new GraphQLList(require('./immunizationperformer.schema.js')),
			description: 'Indicates who performed the immunization event.',
		},
		note: {
			type: new GraphQLList(require('./annotation.schema.js')),
			description:
				'Extra information about the immunization that is not conveyed by the other attributes.',
		},
		reasonCode: {
			type: new GraphQLList(require('./codeableconcept.schema.js')),
			description: 'Reasons why the vaccine was administered.',
		},
		reasonReference: {
			type: new GraphQLList(
				new GraphQLUnionType({
					name: 'ImmunizationreasonReference_reasonReference_Union',
					description:
						'Condition, Observation or DiagnosticReport that supports why the immunization was administered.',
					types: () => [
						require('./condition.schema.js'),
						require('./observation.schema.js'),
						require('./diagnosticreport.schema.js'),
					],
					resolveType(data) {
						if (data && data.resourceType === 'Condition') {
							return require('./condition.schema.js');
						}
						if (data && data.resourceType === 'Observation') {
							return require('./observation.schema.js');
						}
						if (data && data.resourceType === 'DiagnosticReport') {
							return require('./diagnosticreport.schema.js');
						}
					},
				}),
			),
			description:
				'Condition, Observation or DiagnosticReport that supports why the immunization was administered.',
		},
		_isSubpotent: {
			type: require('./element.schema.js'),
			description:
				'Indication if a dose is considered to be subpotent. By default, a dose should be considered to be potent.',
		},
		isSubpotent: {
			type: GraphQLBoolean,
			description:
				'Indication if a dose is considered to be subpotent. By default, a dose should be considered to be potent.',
		},
		subpotentReason: {
			type: new GraphQLList(require('./codeableconcept.schema.js')),
			description: 'Reason why a dose is considered to be subpotent.',
		},
		education: {
			type: new GraphQLList(require('./immunizationeducation.schema.js')),
			description:
				'Educational material presented to the patient (or guardian) at the time of vaccine administration.',
		},
		programEligibility: {
			type: new GraphQLList(require('./codeableconcept.schema.js')),
			description: "Indicates a patient's eligibility for a funding program.",
		},
		fundingSource: {
			type: require('./codeableconcept.schema.js'),
			description:
				'Indicates the source of the vaccine actually administered. This may be different than the patient eligibility (e.g. the patient may be eligible for a publically purchased vaccine but due to inventory issues, vaccine purchased with private funds was actually administered).',
		},
		reaction: {
			type: new GraphQLList(require('./immunizationreaction.schema.js')),
			description:
				'Categorical data indicating that an adverse event is associated in time to an immunization.',
		},
		protocolApplied: {
			type: new GraphQLList(require('./immunizationprotocolapplied.schema.js')),
			description:
				'The protocol (set of recommendations) being followed by the provider who administered the dose.',
		},
	}),
});
