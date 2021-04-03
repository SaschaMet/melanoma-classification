const {
	GraphQLString,
	GraphQLList,
	GraphQLInputObjectType,
} = require('graphql');

/**
 * @name exports
 * @summary NutritionOrderenteralFormula Input Schema
 */
module.exports = new GraphQLInputObjectType({
	name: 'NutritionOrderenteralFormula_Input',
	description: '',
	fields: () => ({
		_id: {
			type: require('./element.input.js'),
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		id: {
			type: GraphQLString,
			description:
				'Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.',
		},
		extension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				'May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and manageable, there is a strict set of governance  applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.',
		},
		modifierExtension: {
			type: new GraphQLList(require('./extension.input.js')),
			description:
				"May be used to represent additional information that is not part of the basic definition of the element and that modifies the understanding of the element in which it is contained and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and manageable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.  Modifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
		},
		baseFormulaType: {
			type: require('./codeableconcept.input.js'),
			description:
				'The type of enteral or infant formula such as an adult standard formula with fiber or a soy-based infant formula.',
		},
		_baseFormulaProductName: {
			type: require('./element.input.js'),
			description:
				"The product or brand name of the enteral or infant formula product such as 'ACME Adult Standard Formula'.",
		},
		baseFormulaProductName: {
			type: GraphQLString,
			description:
				"The product or brand name of the enteral or infant formula product such as 'ACME Adult Standard Formula'.",
		},
		additiveType: {
			type: require('./codeableconcept.input.js'),
			description:
				'Indicates the type of modular component such as protein, carbohydrate, fat or fiber to be provided in addition to or mixed with the base formula.',
		},
		_additiveProductName: {
			type: require('./element.input.js'),
			description:
				'The product or brand name of the type of modular component to be added to the formula.',
		},
		additiveProductName: {
			type: GraphQLString,
			description:
				'The product or brand name of the type of modular component to be added to the formula.',
		},
		caloricDensity: {
			type: require('./quantity.input.js'),
			description:
				'The amount of energy (calories) that the formula should provide per specified volume, typically per mL or fluid oz.  For example, an infant may require a formula that provides 24 calories per fluid ounce or an adult may require an enteral formula that provides 1.5 calorie/mL.',
		},
		routeofAdministration: {
			type: require('./codeableconcept.input.js'),
			description:
				"The route or physiological path of administration into the patient's gastrointestinal  tract for purposes of providing the formula feeding, e.g. nasogastric tube.",
		},
		administration: {
			type: new GraphQLList(
				require('./nutritionorderenteralformulaadministration.input.js'),
			),
			description:
				'Formula administration instructions as structured data.  This repeating structure allows for changing the administration rate or volume over time for both bolus and continuous feeding.  An example of this would be an instruction to increase the rate of continuous feeding every 2 hours.',
		},
		maxVolumeToDeliver: {
			type: require('./quantity.input.js'),
			description:
				'The maximum total quantity of formula that may be administered to a subject over the period of time, e.g. 1440 mL over 24 hours.',
		},
		_administrationInstruction: {
			type: require('./element.input.js'),
			description:
				'Free text formula administration, feeding instructions or additional instructions or information.',
		},
		administrationInstruction: {
			type: GraphQLString,
			description:
				'Free text formula administration, feeding instructions or additional instructions or information.',
		},
	}),
});
